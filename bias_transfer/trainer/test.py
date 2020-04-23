import copy

from bias_transfer.trainer.main_loop import main_loop
from bias_transfer.utils import stringify
from nnvision.utility.measures import get_poisson_loss, get_correlations


def test_neural_model(model, data_loader, device, epoch):
    loss = get_poisson_loss(model, data_loader, device, as_dict=False, per_neuron=False)
    eval = get_correlations(
        model, data_loader, device=device, as_dict=False, per_neuron=False
    )
    print("Test Epoch {}: eval={}, loss={}".format(epoch, eval, loss))
    return eval, loss


def test_model(
    model,
    epoch,
    n_iterations,
    criterion,
    device,
    data_loader,
    config,
    seed,
    noise_test: bool = True,
    eval_type="Validation",
):
    if config.noise_test and noise_test:
        test_eval = {}
        test_loss = {}
        for n_type, n_vals in config.noise_test.items():
            test_eval[n_type] = {}
            test_loss[n_type] = {}
            for val in n_vals:
                val_str = stringify(val)
                config = copy.deepcopy(config)
                config.noise_snr = None
                config.noise_std = None
                setattr(config, n_type, val)
                main_loop_modules = [
                    globals().get("NoiseAugmentation")(
                        config, device, data_loader, seed
                    )
                ]
                test_eval[n_type][val_str], test_loss[n_type][val_str], _ = main_loop(
                    model,
                    criterion,
                    device,
                    None,
                    eval_type=eval_type,
                    data_loader=data_loader,
                    n_iterations=n_iterations,
                    modules=main_loop_modules,
                    train_mode=False,
                    epoch=epoch,
                    neural_prediction=config.neural_prediction,
                )
    else:
        main_loop_modules = []
        for k in config.main_loop_modules:
            if k != "NoiseAugmentation":
                main_loop_modules.append(
                    globals().get(k)(config, device, data_loader, seed)
                )
        test_eval, test_loss, _ = main_loop(
            model,
            criterion,
            device,
            None,
            eval_type=eval_type,
            data_loader=data_loader,
            n_iterations=n_iterations,
            modules=main_loop_modules,
            train_mode=False,
            epoch=epoch,
            neural_prediction=config.neural_prediction,
        )
    return test_eval, test_loss
