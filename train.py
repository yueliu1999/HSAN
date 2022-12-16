from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import hard_sample_aware_network


if __name__ == '__main__':

    # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    for dataset_name in ["cora"]:

        # setup hyper-parameter
        args = setup_args(dataset_name)

        # record results
        file = open("result.csv", "a+")
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for args.seed in range(args.runs):
            # record results

            # fix the random seed
            setup_seed(args.seed)

            # load graph data
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            # apply the laplacian filtering
            X_filtered = laplacian_filtering(A, X, args.t)

            # test
            args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X_filtered, y, cluster_num)

            # build our hard sample aware network
            HSAN = hard_sample_aware_network(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num)

            # adam optimizer
            optimizer = optim.Adam(HSAN.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            # load data to device
            A, HSAN, X_filtered, mask = map(lambda x: x.to(args.device), (A, HSAN, X_filtered, mask))

            # training
            for epoch in tqdm(range(400), desc="training..."):
                # train mode
                HSAN.train()

                # encoding with Eq. (3)-(5)
                Z1, Z2, E1, E2 = HSAN(X_filtered, A)

                # calculate comprehensive similarity by Eq. (6)
                S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)

                # calculate hard sample aware contrastive loss by Eq. (10)-(11)
                loss = hard_sample_aware_infoNCE(S, mask, HSAN.pos_neg_weight, HSAN.pos_weight, node_num)

                # optimization
                loss.backward()
                optimizer.step()

                # testing and update weights of sample pairs
                if epoch % 10 == 0:
                    # evaluation mode
                    HSAN.eval()

                    # encoding
                    Z1, Z2, E1, E2 = HSAN(X_filtered, A)

                    # calculate comprehensive similarity by Eq. (6)
                    S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)

                    # fusion and testing
                    Z = (Z1 + Z2) / 2
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    # select high confidence samples
                    H, H_mat = high_confidence(Z, center)

                    # calculate new weight of sample pair by Eq. (9)
                    M, M_mat = pseudo_matrix(P, S, node_num)

                    # update weight
                    HSAN.pos_weight[H] = M[H].data
                    HSAN.pos_neg_weight[H_mat] = M_mat[H_mat].data

                    # recording
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open("result.csv", "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        file = open("result.csv", "a+")
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
