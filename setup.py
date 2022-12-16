from opt import args


def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"
    args.acc = args.nmi = args.ari = args.f1 = 0

    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1

    elif args.dataset == 'citeseer':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'sigmoid'
        args.tao = 0.3
        args.beta = 2

    elif args.dataset == 'amap':
        args.t = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 3

    elif args.dataset == 'bat':
        args.t = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.3
        args.beta = 5

    elif args.dataset == 'eat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.7
        args.beta = 5

    elif args.dataset == 'uat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500
        args.activate = 'sigmoid'
        args.tao = 0.7
        args.beta = 5

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("confidence: {}".format(args.tao))
    print("focusing factor: {}".format(args.beta))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args
