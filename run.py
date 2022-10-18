import time
from utils import *
from model import SeeGera_Independent, SeeGera_Correlated


def main(args):
    LOSS_NAN = False
    if args.wandb:
        import wandb
        wandb.init(project="SeeGera", entity="xxx")
        wandb.watch_called = False
        config = wandb.config
        config.update(args)

    attr_inference = args.attr_inference
    link_prediction = args.link_prediction
    node_classification = args.node_classification
    assert attr_inference ** 2 + link_prediction ** 2 + node_classification ** 2 == 1
    print(f"Using {args.dataset} dataset")
    seeds = args.seeds
    print(f"seeds = {args.seeds}")
    test_roc_over_runs = []
    test_ap_over_runs = []
    val_acc_over_runs = []
    test_acc_over_runs = []
    val_mse_over_runs = []
    test_mse_over_runs = []
    for i, seed in enumerate(seeds):
        print(f"########## Run {i} for seed {seed} ##########")
        set_random_seed(seed=seed)
        if link_prediction or node_classification:
            adj, features, labels, train_mask, val_mask, test_mask = load_data_with_labels(dataset_str=args.dataset)
        else:
            adj, features, labels, train_mask, val_mask, test_mask = load_data_with_labels_new(dataset_str=args.dataset)
        num_nodes, num_features = features.shape

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        features_orig = features

        if link_prediction:
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
            fea_train = features
        elif attr_inference:
            fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)
            adj_train = adj
        else:
            assert args.node_classification
            fea_train = features
            adj_train = adj

        adj_norm_test, pos_weight_test, norm_test, features_test, pos_weight_a_test, norm_a_test = prepare_inputs(
            adj=adj_train, features=fea_train)
        adj_norm_test = adj_norm_test.to(args.device)
        features_test = features_test.to(args.device)

        adj_label = adj_train + sp.eye(adj_train.shape[0])  # self-loop
        adj_label = torch.FloatTensor(adj_label.toarray()).to(args.device)
        labels = torch.argmax(torch.tensor(labels), dim=1).to(args.device)

        tolerance = 0  # early stopping

        best_roc_val = 0
        best_ap_val = 0
        best_roc_test = 0
        best_ap_test = 0

        best_val_mse = 1e8
        best_test_mse = 1e8

        outer_best_val_acc = 0
        outer_best_epoch = 0
        outer_best_test_acc = 0

        set_random_seed(seed=seed)

        if args.version == 'independent':
            model = SeeGera_Independent(num_nodes=num_nodes,
                                        input_dim=num_features,
                                        num_hidden=args.num_hidden,
                                        out_dim=args.out_dim,
                                        noise_dim=5,
                                        dropout=args.pretrain_dropout,
                                        K=args.K,
                                        J=args.J,
                                        device=args.device,
                                        encoder_type=args.encoder_type,
                                        encoder_layers=args.encoder_layers,
                                        decoder_type=args.decoder_type,
                                        decoder_layers=args.decoder_layers)
        else:
            assert args.version == 'correlated'
            model = SeeGera_Correlated(num_nodes=num_nodes,
                                       input_dim=num_features,
                                       num_hidden=args.num_hidden,
                                       out_dim=args.out_dim,
                                       noise_dim=5,
                                       dropout=args.pretrain_dropout,
                                       K=args.K,
                                       J=args.J,
                                       device=args.device,
                                       encoder_type=args.encoder_type,
                                       encoder_layers=args.encoder_layers,
                                       decoder_type=args.decoder_type,
                                       decoder_layers=args.decoder_layers)

        model.to(args.device)
        optimizer = optim.Adam(params=model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
        for epoch in range(1, args.pretrain_epochs + 1):
            if tolerance > 100:
                break
            start_time = time.time()
            # Graph augmentation
            adj_train_aug = adj_augment(adj_mat=adj_train, aug_prob=args.aug_e)
            fea_train_aug = attr_augment(attr_mat=fea_train, aug_prob=args.aug_a)

            adj_norm, pos_weight, norm, features, pos_weight_a, norm_a = prepare_inputs(adj=adj_train_aug,
                                                                                        features=fea_train_aug)
            adj_norm = adj_norm.to(args.device)
            features = features.to(args.device)

            warmup = np.min([epoch / 300., 1.])
            model.train()
            optimizer.zero_grad()
            merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
            merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
            reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec = model(x=features,
                                                                                                     adj=adj_norm)

            node_attr_mu = torch.cat((merged_node_mu, merged_attr_mu), 0)
            node_attr_sigma = torch.cat((merged_node_sigma, merged_attr_sigma), 0)
            node_attr_z_samples = torch.cat((merged_node_z_samples, merged_attr_z_samples), 0)
            node_attr_logv_iw = torch.cat((node_logv_iw, attr_logv_iw), 0)

            ker = torch.exp(
                -0.5 * (torch.sum(
                    torch.square(node_attr_z_samples - node_attr_mu) / torch.square(node_attr_sigma + args.eps), 3)))

            log_H_iw_vec = torch.log(torch.mean(ker, 2) + args.eps) - 0.5 * torch.sum(node_attr_logv_iw, 2)
            log_H_iw = torch.mean(log_H_iw_vec, 0)

            adj_orig_tile = adj_label.unsqueeze(-1).expand(-1, -1, args.K)  # adj matrix
            log_lik_iw_node = -1 * get_rec_loss(norm=norm,
                                                pos_weight=pos_weight,
                                                pred=reconstruct_node_logits,
                                                labels=adj_orig_tile,
                                                loss_type=args.node_loss_type)

            node_log_prior_iw_vec = -0.5 * torch.sum(torch.square(node_z_samples_iw), 2)
            node_log_prior_iw = torch.mean(node_log_prior_iw_vec, 0)

            features_tile = features.unsqueeze(-1).expand(-1, -1, args.K)  # feature matrix
            log_lik_iw_attr = -1 * get_rec_loss(norm=norm_a,
                                                pos_weight=pos_weight_a,
                                                pred=reconstruct_attr_logits,
                                                labels=features_tile,
                                                loss_type=args.attr_loss_type)

            attr_log_prior_iw_vec = -0.5 * torch.sum(torch.square(attr_z_samples_iw), 2)
            attr_log_prior_iw = torch.mean(attr_log_prior_iw_vec, 0)

            loss = - torch.logsumexp(
                log_lik_iw_node +
                log_lik_iw_attr +
                node_log_prior_iw * warmup / num_nodes +
                attr_log_prior_iw * warmup / num_features -
                log_H_iw * warmup / (num_nodes + num_features), dim=0) + np.log(args.K)
            if torch.isnan(loss):
                LOSS_NAN = True
                break
            loss.backward()
            if epoch % args.display_step == 0:
                print(time.time() - start_time)
                print("Epoch:", '%04d' % epoch, "cost_train=", "{:.9f}".format(loss.item()))
            optimizer.step()

            threshold = 0
            if args.dataset == 'pubmed' and node_classification:
                threshold = 800

            if epoch > threshold:
                with torch.no_grad():
                    model.eval()
                    merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
                    merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
                    node_mu_iw_vec, attr_mu_iw_vec = model.encode(
                        x=features_test,
                        adj=adj_norm_test)

                if link_prediction:
                    roc_curr_val, ap_curr_val = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                                   edges_pos=val_edges,
                                                                   edges_neg=val_edges_false,
                                                                   adj=adj_orig)

                    roc_curr_test, ap_curr_test = get_roc_score_node(emb=node_mu_iw_vec.detach().cpu().numpy(),
                                                                     edges_pos=test_edges,
                                                                     edges_neg=test_edges_false,
                                                                     adj=adj_orig)

                    print("Epoch:", '%04d' % epoch, "val_ap=", "{:.5f}".format(ap_curr_val))
                    print("Epoch:", '%04d' % epoch, "val_roc=", "{:.5f}".format(roc_curr_val))
                    print("Epoch:", '%04d' % epoch, "test_ap=", "{:.5f}".format(ap_curr_test))
                    print("Epoch:", '%04d' % epoch, "test_roc=", "{:.5f}".format(roc_curr_test))
                    print('--------------------------------')

                    if roc_curr_val > best_roc_val and ap_curr_val > best_ap_val:
                        tolerance = 0
                        best_roc_val = roc_curr_val
                        best_ap_val = ap_curr_val
                        best_roc_test = roc_curr_test
                        best_ap_test = ap_curr_test
                    else:
                        tolerance += 1

                elif attr_inference:
                    with torch.no_grad():
                        reconstruct_node_logits, reconstruct_attr_logits = model.decode(adj=adj_norm_test,
                                                                                        x=features_test,
                                                                                        node_z=node_z_samples_iw,
                                                                                        attr_z=attr_z_samples_iw)
                    val_mse = get_mse_attr(
                        fea_rec=reconstruct_attr_logits,
                        feas_pos=val_feas,
                        feas_neg=val_feas_false,
                        features_orig=features_orig)

                    test_mse = get_mse_attr(
                        fea_rec=reconstruct_attr_logits,
                        feas_pos=test_feas,
                        feas_neg=test_feas_false,
                        features_orig=features_orig)

                    if val_mse < best_val_mse:
                        tolerance = 0
                        best_val_mse = val_mse
                        best_test_mse = test_mse
                    else:
                        tolerance += 1

                    print("Epoch:", '%04d' % epoch, "val_mse=", "{:.5f}".format(val_mse))
                    print("Epoch:", '%04d' % epoch, "test_mse=", "{:.5f}".format(test_mse))
                    print("Best val mse=", "{:.5f}".format(best_val_mse))
                    print("Best test mse=", "{:.5f}".format(best_test_mse))
                    print('--------------------------------')

                elif args.node_classification and epoch % args.finetune_interval == 0:
                    final_test_acc, inner_best_val_acc, inner_best_test_acc = node_classification_evaluation(
                        data=node_mu_iw_vec,
                        labels=labels,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask,
                        args=args)
                    print(f"Pretrain epoch {epoch}")
                    # print(
                    #     f"[Inner] --- Best ValAcc: {inner_best_val_acc:.4f}, ",
                    #     f"Final TestAcc: {final_test_acc:.4f}, ",
                    #     f"Best TestAcc: {inner_best_test_acc:.4f} --- ")
                    if inner_best_val_acc > outer_best_val_acc:
                        tolerance = 0
                        outer_best_val_acc = inner_best_val_acc
                        outer_best_epoch = epoch
                        outer_best_test_acc = inner_best_test_acc
                    else:
                        tolerance += 1
                    print(
                        f"[Outer] --- Best ValAcc: {outer_best_val_acc:.4f} in epoch {outer_best_epoch}, ",
                        f"Best TestAcc: {outer_best_test_acc:.4f} --- ")
        if link_prediction:
            print("val_roc:", '{:.5f}'.format(best_roc_val), "val_ap=", "{:.5f}".format(best_ap_val))
            print("test_roc:", '{:.5f}'.format(best_roc_test), "test_ap=", "{:.5f}".format(best_ap_test))
            test_roc_over_runs.append(best_roc_test)
            test_ap_over_runs.append(best_ap_test)
        elif attr_inference:
            print("Val mse:", "{:.5f}".format(best_val_mse))
            print("Test mse:", "{:.5f}".format(best_test_mse))
            val_mse_over_runs.append(best_val_mse)
            test_mse_over_runs.append(best_test_mse)
        else:
            assert node_classification
            if LOSS_NAN:
                break
            print("Node classification, val_acc", '{:.5f}'.format(outer_best_val_acc), "test_acc",
                  '{:.5f}'.format(outer_best_test_acc))
            val_acc_over_runs.append(outer_best_val_acc)
            test_acc_over_runs.append(outer_best_test_acc)
    if link_prediction:
        print("Test ROC", test_roc_over_runs)
        print("Test AP", test_ap_over_runs)
        print("ROC: {:.5f}".format(np.mean(test_roc_over_runs)), "+-", "{:.5f}".format(np.std(test_roc_over_runs)))
        print("AP: {:.5f}".format(np.mean(test_ap_over_runs)), "+-", "{:.5f}".format(np.std(test_ap_over_runs)))

        with open(f'./{args.filename}', 'a') as f:
            f.write(f"pretrain_lr {args.pretrain_lr}, "
                    f"finetune_lr {args.finetune_lr}, "
                    f"pretrain_wd {args.pretrain_wd}, "
                    f"finetune_wd {args.finetune_wd}, "
                    f"pretrain_dropout {args.pretrain_dropout}, "
                    f"finetune_dropout {args.finetune_dropout}, "
                    f"encoder_layers {args.encoder_layers}, "
                    f"decoder_layers {args.decoder_layers}, "
                    f"aug_e {args.aug_e}, "
                    f"aug_a {args.aug_a}\n")
            f.write(f"Test ROC: {np.mean(test_roc_over_runs):.4f}/{np.std(test_roc_over_runs):.4f}\n")
            f.write(f"Test AP: {np.mean(test_ap_over_runs):.4f}/{np.std(test_ap_over_runs):.4f}\n")

        if args.wandb:
            summary = {
                'ROC': np.mean(test_roc_over_runs),
                'ROCStd': np.std(test_roc_over_runs),
                'AP': np.mean(test_ap_over_runs),
                'APStd': np.std(test_ap_over_runs)
            }
            wandb.log(summary)
    elif attr_inference:
        with open(f'./{args.filename}', 'a') as f:
            f.write(f"pretrain_lr {args.pretrain_lr}, "
                    f"finetune_lr {args.finetune_lr}, "
                    f"pretrain_wd {args.pretrain_wd}, "
                    f"finetune_wd {args.finetune_wd}, "
                    f"pretrain_dropout {args.pretrain_dropout}, "
                    f"finetune_dropout {args.finetune_dropout}, "
                    f"encoder_layers {args.encoder_layers}, "
                    f"decoder_layers {args.decoder_layers}, "
                    f"aug_e {args.aug_e}, "
                    f"aug_a {args.aug_a}\n")
            f.write(f"Val mse: {np.mean(val_mse_over_runs)}/{np.std(val_mse_over_runs)}\n")
            f.write(f"Test mse: {np.mean(test_mse_over_runs)}/{np.std(test_mse_over_runs)}\n")
        if args.wandb:
            summary = {
                'ValMse': np.mean(val_mse_over_runs),
                'ValMseStd': np.std(val_mse_over_runs),
                'TestMse': np.mean(test_mse_over_runs),
                'TestMseStd': np.std(test_mse_over_runs)
            }
            wandb.log(summary)
    else:
        assert node_classification
        if LOSS_NAN:
            with open(f'./{args.filename}', 'a') as f:
                f.write(f"pretrain_lr {args.pretrain_lr}, "
                        f"finetune_lr {args.finetune_lr}, "
                        f"pretrain_wd {args.pretrain_wd}, "
                        f"finetune_wd {args.finetune_wd}, "
                        f"pretrain_dropout {args.pretrain_dropout}, "
                        f"finetune_dropout {args.finetune_dropout}, "
                        f"encoder_layers {args.encoder_layers}, "
                        f"decoder_layers {args.decoder_layers}, "
                        f"aug_e {args.aug_e}, "
                        f"aug_a {args.aug_a}\n")
                f.write(f"LOSS NAN\n")
            exit(0)
        print("Val")
        print("Val accuracy", val_acc_over_runs)
        print("Val accuracy: {:.5f}".format(np.mean(val_acc_over_runs)), "+-",
              "{:.5f}".format(np.std(val_acc_over_runs)))

        print("Test")
        print("Test accuracy", test_acc_over_runs)
        print("Test accuracy: {:.5f}".format(np.mean(test_acc_over_runs)), "+-",
              "{:.5f}".format(np.std(test_acc_over_runs)))
        with open(f'./{args.filename}', 'a') as f:
            f.write(f"pretrain_lr {args.pretrain_lr}, "
                    f"finetune_lr {args.finetune_lr}, "
                    f"pretrain_wd {args.pretrain_wd}, "
                    f"finetune_wd {args.finetune_wd}, "
                    f"pretrain_dropout {args.pretrain_dropout}, "
                    f"finetune_dropout {args.finetune_dropout}, "
                    f"encoder_layers {args.encoder_layers}, "
                    f"decoder_layers {args.decoder_layers}, "
                    f"aug_e {args.aug_e}, "
                    f"aug_a {args.aug_a}\n")
            f.write(f"Val accuracy: {np.mean(val_acc_over_runs):.4f}/{np.std(val_acc_over_runs):.4f}\n")
            f.write(f"Test accuracy: {np.mean(test_acc_over_runs):.4f}/{np.std(test_acc_over_runs):.4f}\n")

        if args.wandb:
            summary = {
                'ValAcc': np.mean(val_acc_over_runs),
                'ValStd': np.std(val_acc_over_runs),
                'TestAcc': np.mean(test_acc_over_runs),
                'TestStd': np.std(test_acc_over_runs)
            }
            wandb.log(summary)

            from datetime import timedelta
            from wandb import AlertLevel

            if np.mean(test_acc_over_runs) > args.threshold:
                wandb.alert(
                    title='High accuracy',
                    text=f'Accuracy {np.mean(test_acc_over_runs)} is above the acceptable threshold {args.threshold}',
                    level=AlertLevel.WARN,
                    wait_duration=timedelta(minutes=5)
                )


if __name__ == '__main__':
    args = get_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
