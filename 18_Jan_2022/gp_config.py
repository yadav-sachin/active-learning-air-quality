class GPConfig:
    # Model Training
    n_epochs = 250
    lr = 1e-3
    train_batch_size = 256

    # Laplace parameters
    task_type = "regression"
    subset_of_weights = "all"
    hessian_structure = "full"

    la_noise_n_epochs = 1000
    la_noise_lr = 0.1

    # Gaussian Parameters
    gp_lr = 0.01
    gp_n_epochs = 220  # after early-stopping

    init_random_seed = 43593475
    torch_model_manual_seed = 5498597
