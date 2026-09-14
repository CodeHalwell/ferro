"""Optional training orchestration with explicit update ownership."""
from contextlib import contextmanager, nullcontext
from numbers import Real
from .progress import Progress
from math import prod
from . import Tensor
from .nn import Module


def _report(value):
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise TypeError('Step and metrics callbacks must return dict[str, object]')
    return value


@contextmanager
def _mode_scope(model, training):
    modules = list(model.modules())
    modes = [(module, module.training, module._structure_locked) for module in modules]
    try:
        for module in modules:
            module._structure_locked = True
            module.training = training
        yield
    finally:
        for module, mode, locked in modes:
            module.training = mode
            module._structure_locked = locked


class Trainer:
    def __init__(self, model, *, optimizer=None, loss_fn=None, objective=None,
                 train_step=None, eval_step=None, metrics=None, progress=None,
                 progress_metrics=None, progress_weight=None):
        if not isinstance(model, Module):
            raise TypeError('model must be a Module')
        if sum(x is not None for x in (loss_fn, objective, train_step)) > 1:
            raise ValueError('loss_fn, objective, and train_step are mutually exclusive')
        if train_step is not None and (optimizer is not None or metrics is not None):
            raise ValueError('Custom train_step owns optimizers and metrics')
        if (loss_fn is not None or objective is not None) and optimizer is None:
            raise ValueError('Managed training requires an optimizer')
        for callback in (loss_fn, objective, train_step, eval_step, metrics):
            if callback is not None and not callable(callback):
                raise TypeError('Strategies and metrics must be callable')
        if optimizer is not None and not all(callable(getattr(optimizer, key, None)) for key in ('zero_grad', 'step')):
            raise TypeError('optimizer must provide zero_grad() and step()')
        if progress is not None and not isinstance(progress, (Progress, bool)):
            raise TypeError("progress must be a Progress, bool, or None")
        for callback in (progress_metrics, progress_weight):
            if callback is not None and not callable(callback):
                raise TypeError("Progress adapters must be callable")
        self.progress = Progress() if progress is True else (progress if isinstance(progress, Progress) else None)
        self.progress_metrics, self.progress_weight = progress_metrics, progress_weight
        self.model, self.optimizer = model, optimizer
        self.loss_fn, self.objective = loss_fn, objective
        self.train_step, self.eval_step, self.metrics = train_step, eval_step, metrics

    def _managed(self, batch):
        if self.objective is not None:
            loss = self.objective(self.model, batch)
        else:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise TypeError('Supervised batches must be (inputs, targets)')
            loss = self.loss_fn(self.model(batch[0]), batch[1])
        if not isinstance(loss, Tensor) or prod(loss.shape) != 1:
            raise ValueError('loss/objective must return a scalar Tensor')
        report = {'loss': loss}
        if self.metrics is not None:
            extra = _report(self.metrics(self.model, batch, loss))
            if 'loss' in extra:
                raise ValueError('metrics may not overwrite reserved loss')
            report.update(extra)
        return report

    def _progress_report(self, batch, report):
        if self.progress is None:
            return
        values = (self.progress_metrics(batch, report) if self.progress_metrics is not None
                  else {name: value for name, value in report.items() if isinstance(value, Real)})
        if self.progress_metrics is None and self.progress.tensor_metrics and self.progress.should_log:
            values.update({name: value.item() for name, value in report.items()
                           if isinstance(value, Tensor) and prod(value.shape) == 1})
        weight = self.progress_weight(batch) if self.progress_weight is not None else 1
        self.progress.log(_report(values), weight=weight)

    def fit(self, batches, *, epochs=1):
        from . import enable_grad
        if self.train_step is None and self.loss_fn is None and self.objective is None:
            raise ValueError('fit requires a training strategy')
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError('epochs must be a positive integer')
        if epochs > 1 and iter(batches) is batches:
            raise ValueError('Multiple epochs require a reiterable source, not a one-shot iterator')
        reports = []
        with _mode_scope(self.model, True):
            for _ in range(epochs):
                with self.progress.epoch() if self.progress is not None else nullcontext():
                    source = self.progress.batches(batches) if self.progress is not None else batches
                    for batch in source:
                        if self.train_step is not None:
                            report = _report(self.train_step(self.model, batch))
                        else:
                            with enable_grad():
                                self.optimizer.zero_grad()
                                report = self._managed(batch)
                                if not report['loss'].requires_grad:
                                    raise ValueError('Managed training requires a differentiable loss (requires_grad=True)')
                                report['loss'].backward()
                                self.optimizer.step()
                            if self.progress is not None:
                                self.progress.update()
                        self._progress_report(batch, report)
                        reports.append(report)
        return reports

    def evaluate(self, batches, *, grad_enabled=False):
        from . import no_grad, enable_grad
        if not isinstance(grad_enabled, bool):
            raise TypeError('grad_enabled must be bool')
        if self.eval_step is None and self.loss_fn is None and self.objective is None:
            raise ValueError('evaluate requires eval_step or a managed loss/objective')
        reports = []
        with _mode_scope(self.model, False), (enable_grad() if grad_enabled else no_grad()):
            with self.progress.epoch(phase='validation') if self.progress is not None else nullcontext():
                source = self.progress.batches(batches) if self.progress is not None else batches
                for batch in source:
                    report = _report(self.eval_step(self.model, batch)) if self.eval_step is not None else self._managed(batch)
                    self._progress_report(batch, report)
                    reports.append(report)
        return reports
