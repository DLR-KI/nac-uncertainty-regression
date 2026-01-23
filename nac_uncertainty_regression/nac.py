# SPDX-FileCopyrightText: 2026 DLR e.V.
#
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
import torch
from enum import Enum, auto

from torch._C import device

torch.set_num_interop_threads(8)
torch.set_num_threads(16)

def recursive_getattr(obj: object, key: str) -> object:
    """Used to get nested attributes in dot-notation, e.g. "model.layer1.block3.fc"

    Args:
        obj (object): Object to get attribute from
        key (str): name of the attribute(s) in dot notation

    Returns:
        object: the desired attribute
    """
    obj_ = obj
    for sub_key in key.split("."):
        obj_ = getattr(obj_, sub_key)

    return obj_

def recursive_setattr(obj: object, key: str, value: object):
    """Used to set nested attributes in dot-notation, e.g. "model.layer1.block3.fc"

    Args:
        obj (object): Object to get attribute from
        key (str): name of the attribute(s) in dot notation
        value (object): Value to be set as the attribute

    """
    obj_ = obj
    for sub_key in key.split(".")[:-1]:
        obj_ = getattr(obj_, sub_key)

    setattr(obj_, key.split(".")[-1], value)

class NACMode(Enum):
    """
    Enum controlling the mode NAC should be used in. Currently only "classification" and "regression" are supported.
    """
    CLASSIFICATION = auto()
    REGRESSION = auto()


@torch.jit.script
def _compute_loss_classification(network_output: torch.Tensor, network_output_key: str | None, device: device,
                                 class_dimension: int = -1) -> torch.Tensor:
    """Computes the kl divergence between an uniform class output vector and the actual softmax of the network output.
    As we're only interested in the gradients of the loss later on, the constant part of the formula is omitted!

    Args:
        network_output (torch.Tensor): final logits of network, shape (batch_size, n_classes)
        network_output_key (str | None): key that identifies the relevant network output if network outputs a dict, ignored otherwise
        device (device): Ignored, only here for API consistency
        class_dimension (int, optional): index of the class logits. Defaults to -1.

    Returns:
        torch.Tensor: KL Divergence value, shape (batch_size,)
    """
    if isinstance(network_output, torch.Tensor):
        n_classes = network_output.shape[class_dimension]
    else:
        assert isinstance(network_output, dict), "Only tensor and dict are supported!"
        n_classes = network_output[network_output_key].shape[class_dimension]
    if n_classes > 1:
        uniform_label = 1 / n_classes
        kl_div = - uniform_label * torch.sum(torch.nn.functional.log_softmax(network_output, dim=-1), dim=-1) # type: ignore


    else:
        kl_div = - 0.5 * (torch.nn.functional.logsigmoid(network_output) + torch.log(1 - torch.sigmoid(network_output)))        # a single output neuron should output 0.5 if maximally unsure


    return kl_div


@torch.jit.script
def _compute_loss_regression(network_output: torch.Tensor, n: int, sum: torch.Tensor, sum_squares: torch.Tensor,
                             device: device) -> torch.Tensor:
    """Computes an experimental loss for regression problems, by taking the mahalanobis distance between the mean network output and the outputs

    Args:
        network_output (torch.Tensor): final logits of network, shape (batch_size, n_outputs)
        n (int): number of data points for running mean and std
        sum (torch.Tensor): sum for running mean and std
        sum_squares (torch.Tensor): sum of squares for runnign mean and std
        device (device): device to compute on

    Returns:
        torch.Tensor: mahalanobis distance between network output and distribution mean
    """
    mean_output = torch.as_tensor(sum / n, device=device) # type: ignore
    std_deviation = torch.sqrt(sum_squares / n - torch.square(torch.as_tensor(sum / n, device=device))) # type: ignore
    output_distance_from_mean_normed = ((network_output - mean_output[None, ...]) / (std_deviation[None, ...] + 1e-8)) ** 2 # type: ignore
    return output_distance_from_mean_normed


@torch.jit.script
def _compute_neuron_activation_states(network_output: torch.Tensor, layer_activations: dict[str, torch.Tensor],
                                      stats_n: int, stats_sum: torch.Tensor, stats_sum_squares: torch.Tensor,
                                      layers_to_monitor: list[str], device: device, alpha: float = 100,
                                      network_output_key: str | None = None, mode: NACMode = NACMode.CLASSIFICATION,
                                      class_dimension: int = -1) -> dict[str, torch.Tensor]:
    """Given the final network output and the corresponding activations of intermediate layers, this method computes the activations tates of the neurons
    by backpropagating from the KLdivergence between the network output and a uniform vector.

    Args:
        network_output (torch.Tensor): final logits of network, shape (batch_size, n_classes)
        layer_activations (dict[str, torch.Tensor]): dict of (layer_name -> raw activation map / vector)
        stats_n (int): number of data points for running mean and std
        stats_sum (torch.Tensor): sum for running mean and std
        stats_sum_squares (torch.Tensor): sum of squares for runnign mean and std
        layers_to_monitor (list[str]): list of layer names (in dot notation) to compute activation states on
        device (device): device to compute on
        alpha (float, optional): sharpness of sigmoid. Defaults to 100.
        network_output_key (str | None, optional): key that identifies the relevant network output if network outputs a dict, ignored otherwise. Defaults to None.
        mode (NACMode, optional): Use Enum to specify type of ML-problem. Defaults to NACMode.CLASSIFICATION.
        class_dimension (int, optional): index of the class logits. Defaults to -1.

    Raises:
        NotImplementedError: If mode is not classification or regression (shouldn't happen)
        RuntimeError: If Gradient Computation fails for any reason

    Returns:
        dict[str, torch.Tensor]: the states of the neurons as dict of (layer_name -> tensor of neuron states, shape (batch_size, n_neurons))
    """
    if mode == NACMode.CLASSIFICATION:
        loss = _compute_loss_classification(network_output, network_output_key, device, class_dimension)
    elif mode == NACMode.REGRESSION:
        loss = _compute_loss_regression(network_output, stats_n, stats_sum, stats_sum_squares, device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")

    neuron_activation_states = {}

    for name in layers_to_monitor:
        (grad,) = torch.autograd.grad([loss.sum()], [layer_activations[name]], create_graph=False, retain_graph=True)
        if grad is not None:
            neuron_states = torch.sigmoid(layer_activations[name] * grad * alpha).detach()
        else:
            raise RuntimeError("Gradient Computation failed!")
        if len(neuron_states.shape) > 2:
            # average pooling if we are dealing with e.g. a CNN (TODO this may not work for ViT!)
            neuron_states = torch.mean(neuron_states, dim=list(range(2, len(neuron_states.shape)))).detach()
        neuron_activation_states[name] = neuron_states

    return neuron_activation_states

@torch.jit.script
def _update_network_stats(network_output: torch.Tensor, n: int, sum: torch.Tensor, sum_squares: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    update running stats for keeping track of output distribution
    Args:
        network_output (torch.Tensor): output of the neural network
        n (int): number of data points already seen in the running stats
        sum (torch.Tensor): sum of outputs for running stats
        sum_squares (torch.Tensor): sum of squared outputs for running stats

    Returns:
        tuple[int, torch.Tensor, torch.Tensor]: tuple of updated (n, sum, sum_squares) after observing network_output
    """
    if n == 0:
        return network_output.shape[0], network_output.sum(dim=0), torch.square(network_output).sum(dim=0)

    return n + network_output.shape[0], (network_output.sum(dim=0) + sum).detach(), (torch.square(network_output).sum(dim=0) + sum_squares).detach()

@torch.jit.script
def _compute_histogram_updates(network_output: torch.Tensor, layer_activations: dict[str, torch.Tensor],
                      stats_n: int, stats_sum: torch.Tensor, stats_sum_squares: torch.Tensor,
                      layers_to_monitor: list[str], alpha: float, network_output_key: str | None,
                      mode: NACMode, class_dimension: int, histograms: dict[str, torch.Tensor],
                      M: int, device: device) -> dict[str, torch.Tensor]:
    """
    Updates the internal histogram for I.D. data.
    Args:
        network_output (torch.Tensor): Raw logits of network output for the current batch.
        layer_activations (dict[str, torch.Tensor]): Layer activations of the current batch.
        stats_n (int): number of data points already seen in the running stats
        stats_sum (torch.Tensor): sum of outputs for running stats
        stats_sum_squares (torch.Tensor): sum of squared outputs for running stats
        layers_to_monitor (list[str]): list of layer names (in dot notation) to compute activation states on
        alpha (float): sharpness of sigmoid.
        network_output_key (str | None): key that identifies the relevant network output if network outputs a dict, ignored otherwise.
        mode (NACMode): Use Enum to specify type of ML-problem.
        class_dimension (int): index of the class logits.
        histograms (dict[str, torch.Tensor]): dict mapping layer names to the corresponding histograms of activations
        M (int): Hyperparameter controlling the number of histogram bins
        device (device): device to compute on

    Returns:
        dict[str, torch.Tensor]: additive updates to histograms of neuron activation
    """
    neuron_activation_states = _compute_neuron_activation_states(
        network_output=network_output,
        layer_activations=layer_activations,
        stats_n=stats_n,
        stats_sum=stats_sum,
        stats_sum_squares=stats_sum_squares,
        layers_to_monitor=layers_to_monitor,
        alpha=alpha,
        network_output_key=network_output_key,
        mode=mode,
        class_dimension=class_dimension,
        device=device
    )
    histogram_updates: dict[str, torch.Tensor] = dict()
    for name in histograms:
        histogram_updates[name] = torch.zeros((neuron_activation_states[name].shape[1], M), device=device).detach() # type: ignore
        for i_neuron in range(neuron_activation_states[name].shape[1]):
            histogram_updates[name][i_neuron] += torch.histc(neuron_activation_states[name][:, i_neuron], bins=M, min=0, max=1) # type: ignore

    return histogram_updates

@torch.jit.script
def _compute_uncertainty(network_output: torch.Tensor, layer_activations: dict[str, torch.Tensor],
                         stats_n: int, stats_sum: torch.Tensor, stats_sum_squares: torch.Tensor,
                         layers_to_monitor: list[str], alpha: float, network_output_key: str | None,
                         mode: NACMode, class_dimension: int, histograms: dict[str, torch.Tensor],
                         M: int, O: int, device: device) -> torch.Tensor:
    """Computes the instance-wise uncertainty a given network output and corresponding intermediate layer activations.
        This assumes that self.histogram has been fitted!

    Args:
        network_output (torch.Tensor): final logits of network, shape (batch_size, n_classes)
        layer_activations (dict[str, torch.Tensor]): dict of (layer_name -> raw activation map / vector)
        stats_n (int): number of data points already seen in the running stats
        stats_sum (torch.Tensor): sum of outputs for running stats
        stats_sum_squares (torch.Tensor): sum of squared outputs for running stats
        layers_to_monitor (list[str]): list of layer names (in dot notation) to compute activation states on
        alpha (float): sharpness of sigmoid.
        network_output_key (str | None): key that identifies the relevant network output if network outputs a dict, ignored otherwise.
        mode (NACMode): Use Enum to specify type of ML-problem.
        class_dimension (int): index of the class logits.
        histograms (dict[str, torch.Tensor]): dict mapping layer names to the corresponding histograms of activations
        M (int): Hyperparameter controlling the number of histogram bins
        O (int): Hyperparameter controlling the number of data points before one histogram bin is considered "full"
        device (device): device to compute on

    Returns:
        torch.Tensor: Uncertainty tensor (float), shape (batch_size,)
    """
    neuron_activation_states = _compute_neuron_activation_states(
        network_output=network_output,
        layer_activations=layer_activations,
        stats_n=stats_n,
        stats_sum=stats_sum,
        stats_sum_squares=stats_sum_squares,
        layers_to_monitor=layers_to_monitor,
        alpha=alpha,
        network_output_key=network_output_key,
        mode=mode,
        class_dimension=class_dimension,
        device=device
    )
    # one uncertainty value per sample!
    uncertainty: torch.Tensor = torch.zeros(network_output.shape[0], device=device).detach()         # type: ignore
    for name in neuron_activation_states:
        bin_idxs = torch.clamp((neuron_activation_states[name] // (1 / M)).long(), torch.as_tensor(0, device=device),  # type: ignore
                               torch.as_tensor(M - 1, device=device)) # type: ignore
        assert histograms[name] is not None
        # Extract the histogram values at the bin indices for all neurons at once
        values = histograms[name][torch.arange(neuron_activation_states[name].shape[1], device=device), bin_idxs] / O # type: ignore
        # Clamp to max 1
        clamped_values = torch.minimum(torch.ones_like(values, device=device), values) # type: ignore
        # Sum all contributions (instead of .item(), sum all)
        layer_score = clamped_values.sum(dim=1) / neuron_activation_states[name].shape[1]
        uncertainty += layer_score

    return (1 / uncertainty)  # the standard NAC score is how *common* the activation pattern was, so uncertainty should be the inverse!

@torch.jit.script
def _nac_forward(layer_activations: dict[str, torch.Tensor],
                 net_output: torch.Tensor,
                 training: bool,
                 stats_n: int,
                 stats_sum: torch.Tensor,
                 stats_sum_squares: torch.Tensor,
                 histograms: dict[str, torch.Tensor],
                 layers_to_monitor: list[str],
                 alpha: float,
                 network_output_key: str | None,
                 mode: NACMode,
                 class_dimension: int,
                 M: int,
                 O: int,
                 device: device
                 ) -> dict[str, int | torch.Tensor | dict[str, torch.Tensor]]:
    """Entry function for internal NAC calculations. Updates hisograms with I.D. data during configuration phase (training),
    computes uncertainty otherwise.

    Args:
        layer_activations (dict[str, torch.Tensor]): dict of (layer_name -> raw activation map / vector)
        net_output (torch.Tensor): final logits of network, shape (batch_size, n_classes)
        training (bool): _description_
        stats_n (int): number of data points already seen in the running stats
        stats_sum (torch.Tensor): sum of outputs for running stats
        stats_sum_squares (torch.Tensor): sum of squared outputs for running stats
        histograms (dict[str, torch.Tensor]): dict mapping layer names to the corresponding histograms of activations
        layers_to_monitor (list[str]): list of layer names (in dot notation) to compute activation states on
        alpha (float): sharpness of sigmoid.
        network_output_key (str | None): key that identifies the relevant network output if network outputs a dict, ignored otherwise.
        mode (NACMode): Use Enum to specify type of ML-problem.
        class_dimension (int): index of the class logits.
        M (int): Hyperparameter controlling the number of histogram bins
        O (int): Hyperparameter controlling the number of data points before one histogram bin is considered "full"
        device (device): device to compute on

    Returns:
        dict[str, int | torch.Tensor | dict[str, torch.Tensor]]: dict with keys:
            uncertainty: contains uncertainty values if training == False
            histograms: contains (additive) histogram updates if training == True
            stats_n: Contains number of data points for running stats
            stats_sum: Contains sum of activations for running stats
            stats_sum_squares: Contains sum of squares of activations for running stats
    """
    output = torch.zeros(len(net_output))
    if training:
        # for dim_idx in range(net_output_view.shape[1]):
        #     net_output_ = net_output_view[:, dim_idx, :]
        stats_n, stats_sum, stats_sum_squares = _update_network_stats(
            net_output, stats_n, stats_sum, stats_sum_squares
        )
        histogram_update = _compute_histogram_updates(
            network_output=net_output,
            layer_activations=layer_activations,
            stats_n=stats_n,
            stats_sum=stats_sum,
            stats_sum_squares=stats_sum_squares,
            layers_to_monitor=layers_to_monitor,
            alpha=alpha,
            network_output_key=network_output_key,
            mode=mode,
            class_dimension=class_dimension,
            histograms=histograms,
            M=M,
            device=device
        ) # type: ignore

        for name in layers_to_monitor:
            if len(histograms[name].shape) > 0:
                histograms[name] += histogram_update[name]
            else:
                histograms[name] = histogram_update[name]

    else:
        output = _compute_uncertainty( # type: ignore
            network_output=net_output,
            layer_activations=layer_activations,
            stats_n=stats_n,
            stats_sum=stats_sum,
            stats_sum_squares=stats_sum_squares,
            layers_to_monitor=layers_to_monitor,
            alpha=alpha,
            network_output_key=network_output_key,
            mode=mode,
            class_dimension=class_dimension,
            histograms=histograms,
            M=M,
            O=O,
            device=device)

    return dict(
        uncertainty=output,
        histograms=histograms,
        stats_n=stats_n,
        stats_sum=stats_sum,
        stats_sum_squares=stats_sum_squares,
    )


class NACWrapper(torch.nn.Module):

    class ActivationCachingWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, name: str):
            """Caches the activations of the provided torch.nn.Module
            Usage:
            model = ResNet50(...)
            model.layer1 = ActivationCachingWrapper(model.layer1)   # now layer1 gets cached
            model(input_data)
            print(model.layer1.pop_all())       # retrieve last activations and empty the cache
            Args:
                module (torch.nn.Module): the module to be wrapped
                name (str): name of the layer for debugging and logging purposes
            """
            super().__init__()
            self.cache: list[torch.Tensor] = []     # list as an attribute is not torch-script compatible... We need a different solution!
            self.module = module
            self.name = name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Calls forward on the underlying module and caches the result
            Module output can then be retrieved with .pop_all()
            Args:
                x (torch.Tensor): Input to the layer

            Returns:
                torch.Tensor: the result of forward on the underlying module
            """
            act = self.module(x)
            assert isinstance(act, torch.Tensor), f"Module {self.name} did not return tensor, but {act}!"
            self.cache.append(act)
            return act

        def reset(self) -> None:
            """resets the cache
            """
            self.cache = []

        def pop_all(self) -> list[torch.Tensor]:
            """return the cache and then resets it

            Returns:
                list[torch.Tensor]: cache contents
            """
            tmp = self.cache
            self.reset()
            return tmp

    def __init__(self, model: torch.nn.Module,
                 layer_name_list: list[str],
                 mode: NACMode = NACMode.CLASSIFICATION,
                 O: int = 2000,
                 alpha: float = 100,
                 M: int = 50,
                 confidence_cutoff: float = 0.1,
                 class_dimension: int = -1,
                 network_output_key: str | None = None,
                 device: device | None = torch.device("cpu")) -> None:
        """A Wrapper class to provide uncertainty estimations to trained models by building a distribution of neuron states
        for selected layers over I.D. data. Epistemic uncertainty can then be calculated for unseen data from the distribution.
        Hyperparameter search over [O, alpha, M] should be considered, as the values are strongly dataset-dependant.

        Note that it is assumed that the model is already trained. If raw model access is needed (e.g. for fine-tuning),
        use the context manager NACWrapper.raw_model_access. Note that it would be best to re-fit the Wrapper after changing anything about the model.

        Args:
            model (torch.nn.Module): The torch model to wrap
            layer_name_list (Iterable[str]): list of the attribute names of the layers that should be monitored, e.g. "layer1" in resnet
            mode (NACMode): Indicate if underlying problem is Classification or Regression
            O (int, optional): Number of samples until a bin is considered 'filled'. A good default is (1 / M) * dataset_size. Defaults to 2000.
            alpha (float, optional): Sharpness of Sigmoid function. Defaults to 100.
            M (int, optional): Number of histogram bins. Defaults to 50.
            confidence_cutoff (float): Unused, for future extension.
            class_dimension (int, optional): Index of the class dimension of the network output. Defaults to -1.
            network_output_key (str, Optional): key that identifies the relevant network output if network outputs a dict, ignored otherwise.
            device (device): torch-device to compute on

        Raises:
            RuntimeError: If GradientComputation is disabled
        """
        super().__init__()
        if not torch.is_grad_enabled():
            raise RuntimeError("NACWrapper requires gradient computation to be enabled! Please do not disable gradients!")

        self.layers_to_monitor = layer_name_list
        self._model = model
        self._model.eval()       # we freeze the model to eval mode
        self.class_dimension = class_dimension
        self.O = O
        self.alpha = alpha
        self.M = M
        self.mode = mode
        self.confidence_cutoff = confidence_cutoff
        for name in layer_name_list:
            recursive_setattr(model, name, NACWrapper.ActivationCachingWrapper(recursive_getattr(model, name), name)) # type: ignore

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for name in layer_name_list:
                try:
                    self.device = recursive_getattr(self._model, name).module.device # type: ignore
                except AttributeError:
                    pass
        else:
            self.device = device

        self.histograms: dict[str, torch.Tensor] = {
            layer_name: torch.tensor(0, device=self.device)
            for layer_name in layer_name_list
        }
        self.train()
        self.stats_n = 0
        self.stats_sum: torch.Tensor = torch.tensor(0, device=self.device)
        self.stats_sum_squares: torch.Tensor = torch.tensor(0, device=self.device)
        self.network_output_key = network_output_key


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Calls forward on the wrapped model and computes the uncertainty for eeach sample.
        Result of wrapped model is stored in "out" key. Uncertainty is stored in "uncertainty" key.
        WARNING:
        Uncertainty is only computed if wrapper is in eval() mode.
        In train() mode, the data is implicitly assumed to be in-distribution,
        used for updating the internal uncertainty estimator and the uncertainty is None!

        Args:
            x (torch.Tensor): input data

        Returns:
            dict[str, torch.Tensor | None]: dict with keys "uncertainty" and "out"
        """
        out =  {"out": self._model(x)}
        net_output = out["out"] if self.network_output_key is None else out["out"][self.network_output_key]
        if len(net_output.shape) > 2:
            assert len(net_output.shape) == 3, f"ndim={len(net_output.shape)} is unsupported!"
            net_output_ = torch.max(net_output, dim=1).values
        else:
            net_output_ = net_output
        layer_activations = self._get_layer_activations()
        output_dict = _nac_forward(
            layer_activations=layer_activations,
            net_output=net_output_,
            training=self.training,
            stats_n=self.stats_n, # type: ignore
            stats_sum=self.stats_sum,
            stats_sum_squares=self.stats_sum_squares,
            histograms=self.histograms,
            layers_to_monitor=self.layers_to_monitor,
            alpha=self.alpha,
            network_output_key=self.network_output_key,
            mode=self.mode,
            class_dimension=self.class_dimension,
            M=self.M, O=self.O,
            device=self.device
        )
        self._reset_layer_activations()
        self.histograms = output_dict["histograms"] # type: ignore
        self.stats_n = output_dict["stats_n"]
        self.stats_sum = output_dict["stats_sum"] # type: ignore
        self.stats_sum_squares = output_dict["stats_sum_squares"] # type: ignore
        out["uncertainty"] = output_dict["uncertainty"]
        return out

    def _get_layer_activations_and_reset(self) -> dict[str, torch.Tensor]:
        """Get activations of all specified layers (self.layer_name_list) and reset the caches

        Returns:
            dict[str, torch.Tensor]: dict of (layer_name -> activation as tensor)
        """
        ls = {}
        for name in self.layers_to_monitor:
            wrapped_layer: "NACWrapper.ActivationCachingWrapper" = recursive_getattr(self._model, name) # type: ignore
            # TODO: this is an ugly fix for the detached gradient problem, but we somehow need the list
            ls[name] = wrapped_layer.cache[0]
            wrapped_layer.reset()
        return ls

    def _get_layer_activations(self) -> dict[str, torch.Tensor]:
        """Get activations of all specified layers (self.layer_name_list) and reset the caches

        Returns:
            dict[str, torch.Tensor]: dict of (layer_name -> activation as tensor)
        """
        ls = {}
        for name in self.layers_to_monitor:
            wrapped_layer: "NACWrapper.ActivationCachingWrapper" = recursive_getattr(self._model, name) # type: ignore
            # TODO: this is an ugly fix for the detached gradient problem, but we somehow need the list
            ls[name] = wrapped_layer.cache[0]

        return ls

    def _reset_layer_activations(self) -> None:
        """Reset all cached layer activations
        """
        for name in self.layers_to_monitor:
            wrapped_layer: "NACWrapper.ActivationCachingWrapper" = recursive_getattr(self._model, name) # type: ignore
            wrapped_layer.reset()

    def train(self, mode: bool = True) -> "NACWrapper":
        """Explicitly overwrites the default behavior of the nn.Module.train() method
        so that we can set train() and eval() on the wrapper without affecting self.model

        Args:
            mode (bool, optional): _description_. Defaults to True.

        Returns:
            Self: self
        """
        self.training = mode
        return self # type: ignore

    def eval(self) -> "NACWrapper":
        """Explicitly overwrites the default behavior of the nn.Module.train() method
        so that we can set train() and eval() on the wrapper without affecting self.model

        Returns:
            Self: self
        """
        self.training = False
        return self # type: ignore

    @contextmanager
    def raw_model_access(self, training: bool = True):
        """Use this context manager to access the raw model, e.g. for re-training.
        It ensures that the training state ofg the wrapper and of the model are properly separated.

        Args:
            training (bool, optional): Wether to set the model to train() when returning it. Defaults to True.

        Yields:
            torch.nn.Module: Reference to the raw model.
        """
        try:
            if training:
                self._model.train()
            yield self._model
        finally:
            self._model.eval()
