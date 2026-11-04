"""Learning-rate schedules over native optimizer groups.

Call scheduler.step() after optimizer.step(). Scheduler state is an owning,
Python dictionary; save it alongside the training checkpoint. Construct
the scheduler before restoring optimizer/checkpoint state, then load its state.
Loading a scheduler does not change optimizer rates, matching PyTorch. Dynamic
group addition after scheduler construction and tensor-valued rates are rejected.
"""
import copy
from bisect import bisect_right
import math
import types
import struct
from .optim import _Base, _number
from ._native import Tensor


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


class _State:
    def state_dict(self):
        return copy.deepcopy({key: value for key, value in self.__dict__.items() if key != 'optimizer'})

    def load_state_dict(self, state_dict):
        if type(state_dict) is not dict or set(state_dict) != set(self.state_dict()):
            raise ValueError('scheduler state schema mismatch')
        candidate = object.__new__(type(self))
        candidate.__dict__.update(copy.deepcopy(state_dict))
        candidate.optimizer = self.optimizer
        candidate._validate()
        self.__dict__.update(candidate.state_dict())

    def get_last_lr(self):
        return list(self._last_lr)

    def _optimizer(self, optimizer):
        if not isinstance(optimizer, _Base):
            raise TypeError('scheduler requires a Ferro optimizer')
        optimizer._validate_membership()
        for group in optimizer.param_groups:
            optimizer._validate_group(group)
        self.optimizer = optimizer

    def _rates(self, values, name):
        if type(values) is not list or len(values) != len(self.optimizer.param_groups):
            raise ValueError(f'{name} must match optimizer group count')
        for value in values:
            _number(value, name)

    def _apply(self, values):
        self._rates(values, 'learning rates')
        self.optimizer._validate_membership()
        groups = [{**group, 'lr': rate} for group, rate in zip(self.optimizer.param_groups, values)]
        self.optimizer._native.set_configurations(self.optimizer._validated_configurations(groups))
        for group, rate in zip(self.optimizer.param_groups, values):
            group['lr'] = rate


class LRScheduler(_State):
    def __init__(self, optimizer, last_epoch=-1):
        _integer(last_epoch, 'last_epoch', -1)
        self._optimizer(optimizer)
        if last_epoch != -1 and any('initial_lr' not in group for group in optimizer.param_groups):
            raise ValueError('resuming a scheduler requires initial_lr in every group')
        self.base_lrs = [group.get('initial_lr', group['lr']) for group in optimizer.param_groups]
        self._rates(self.base_lrs, 'base_lrs')
        for group, value in zip(optimizer.param_groups, self.base_lrs):
            group.setdefault('initial_lr', value)
        self.last_epoch, self._step_count = last_epoch, 0
        self._is_initial, self._get_lr_called_within_step = True, False
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        self.step()
        self._is_initial = False

    def _validate(self):
        self.optimizer._validate_membership()
        self._rates(self.base_lrs, 'base_lrs')
        self._rates(self._last_lr, '_last_lr')
        _integer(self.last_epoch, 'last_epoch', -1)
        _integer(self._step_count, '_step_count')
        if type(self._is_initial) is not bool or type(self._get_lr_called_within_step) is not bool:
            raise ValueError('invalid scheduler flags')
        self._validate_options()

    def step(self, epoch=None):
        self._validate()
        if epoch is not None:
            _integer(epoch, 'epoch')
        candidate = copy.copy(self)
        candidate._step_count += 1
        candidate.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        rates = candidate.get_lr() if epoch is None else candidate._closed_form()
        self._apply(rates)
        self.last_epoch, self._step_count, self._last_lr = candidate.last_epoch, candidate._step_count, list(rates)


class LambdaLR(LRScheduler):
    """Scale base rates with reconstructed functions or callable objects.

    State saves callable-object attributes, not executable functions or closure
    environments. Reconstruct compatible callbacks before loading. A callback
    can mutate external state; failed steps do not roll those side effects back.
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self._optimizer(optimizer)
        self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * len(optimizer.param_groups)
        self._validate_options()
        new_initial = [group for group in optimizer.param_groups if 'initial_lr' not in group]
        try:
            super().__init__(optimizer, last_epoch)
        except BaseException:
            for group in new_initial:
                group.pop('initial_lr', None)
            raise

    def _validate_options(self):
        if type(self.lr_lambdas) is not list or len(self.lr_lambdas) != len(self.optimizer.param_groups):
            raise ValueError('lr_lambda must provide one callable per optimizer group')
        for callback in self.lr_lambdas:
            if not callable(callback) or (not isinstance(callback, types.FunctionType) and type(getattr(callback, '__dict__', None)) is not dict):
                raise TypeError('lr_lambda requires Python functions or callable objects with attribute dictionaries')

    def get_lr(self):
        return [base * _number(callback(self.last_epoch), 'lr_lambda factor') for base, callback in zip(self.base_lrs, self.lr_lambdas)]

    def _closed_form(self):
        return self.get_lr()

    def state_dict(self):
        state = copy.deepcopy({key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')})
        state['lr_lambdas'] = [None if isinstance(callback, types.FunctionType) else copy.deepcopy(callback.__dict__) for callback in self.lr_lambdas]
        return state

    def load_state_dict(self, state_dict):
        if type(state_dict) is not dict or set(state_dict) != set(self.state_dict()):
            raise ValueError('scheduler state schema mismatch')
        staged = copy.deepcopy(state_dict)
        callbacks = staged.pop('lr_lambdas')
        if type(callbacks) is not list or len(callbacks) != len(self.lr_lambdas):
            raise ValueError('saved lr_lambdas must match optimizer group count')
        updates = []
        for callback, attributes in zip(self.lr_lambdas, callbacks):
            if isinstance(callback, types.FunctionType):
                if attributes is not None:
                    raise ValueError('saved Python function state must be None')
            else:
                if type(attributes) is not dict or any(type(key) is not str for key in attributes):
                    raise ValueError('saved callable object state must be an attribute dictionary')
                target = callback.__dict__
                if type(target) is not dict:
                    raise ValueError('callable object attributes must be a dictionary')
                updates.append((target, attributes))
        candidate = object.__new__(type(self))
        candidate.__dict__.update(staged)
        candidate.optimizer, candidate.lr_lambdas = self.optimizer, self.lr_lambdas
        candidate._validate()
        for target, attributes in updates:
            target.update(attributes)
        self.__dict__.update(staged)


class MultiplicativeLR(LambdaLR):
    """Scale current rates with callbacks; callback state follows LambdaLR."""
    def get_lr(self):
        if self._is_initial:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * _number(callback(self.last_epoch), 'lr_lambda factor') for group, callback in zip(self.optimizer.param_groups, self.lr_lambdas)]


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _integer(self.step_size, 'step_size', 1)
        _number(self.gamma, 'gamma')

    def get_lr(self):
        rates = [group['lr'] for group in self.optimizer.param_groups]
        return rates if self.last_epoch == 0 or self.last_epoch % self.step_size else [rate * self.gamma for rate in rates]

    def _closed_form(self):
        return [rate * self.gamma ** (self.last_epoch // self.step_size) for rate in self.base_lrs]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=.1, last_epoch=-1):
        self.milestones, self.gamma = list(milestones), gamma
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        if type(self.milestones) is not list:
            raise ValueError('milestones must be a list of nonnegative integers')
        for value in self.milestones:
            _integer(value, 'milestone')
        _number(self.gamma, 'gamma')

    def get_lr(self):
        factor = self.gamma ** self.milestones.count(self.last_epoch)
        return [group['lr'] * factor for group in self.optimizer.param_groups]

    def _closed_form(self):
        factor = self.gamma ** sum(value <= self.last_epoch for value in self.milestones)
        return [rate * factor for rate in self.base_lrs]


class ConstantLR(LRScheduler):
    """Scale initial rates, then restore their scale after total_iters >= 1."""
    def __init__(self, optimizer, factor=1./3, total_iters=5, last_epoch=-1):
        self.factor, self.total_iters = factor, total_iters
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _number(self.factor, 'factor', positive=True)
        if self.factor > 1:
            raise ValueError('factor must be in (0, 1]')
        _integer(self.total_iters, 'total_iters', 1)

    def get_lr(self):
        factor = self.factor if self.last_epoch == 0 else 1./self.factor if self.last_epoch == self.total_iters else 1.
        return [group['lr'] * factor for group in self.optimizer.param_groups]

    def _closed_form(self):
        factor = self.factor if self.last_epoch < self.total_iters else 1.
        return [rate * factor for rate in self.base_lrs]


class LinearLR(LRScheduler):
    def __init__(self, optimizer, start_factor=1./3, end_factor=1., total_iters=5, last_epoch=-1):
        self.start_factor, self.end_factor, self.total_iters = start_factor, end_factor, total_iters
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _number(self.start_factor, 'start_factor', positive=True)
        _number(self.end_factor, 'end_factor')
        if self.start_factor > 1 or self.end_factor > 1:
            raise ValueError('start_factor must be in (0, 1] and end_factor in [0, 1]')
        _integer(self.total_iters, 'total_iters', 1)

    def get_lr(self):
        if self.last_epoch == 0:
            factor = self.start_factor
        elif self._is_initial or self.last_epoch > self.total_iters:
            factor = 1.
        elif self.last_epoch == self.total_iters and self.end_factor == 0.:
            # Evaluate the exact zero endpoint without cancellation producing
            # a tiny negative rate that native optimizers correctly reject.
            factor = 0.
        else:
            delta = self.end_factor - self.start_factor
            factor = 1. + delta / (self.total_iters * self.start_factor + (self.last_epoch - 1) * delta)
        return [group['lr'] * factor for group in self.optimizer.param_groups]

    def _closed_form(self):
        factor = self.start_factor + (self.end_factor-self.start_factor) * min(self.total_iters,self.last_epoch) / self.total_iters
        return [rate * factor for rate in self.base_lrs]


class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1., last_epoch=-1):
        self.total_iters, self.power = total_iters, power
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _integer(self.total_iters, 'total_iters', 1)
        _number(self.power, 'power')

    def get_lr(self):
        if self._is_initial or self.last_epoch > self.total_iters:
            factor = 1.
        else:
            factor = ((1.-self.last_epoch/self.total_iters) / (1.-(self.last_epoch-1)/self.total_iters)) ** self.power
        return [group['lr'] * factor for group in self.optimizer.param_groups]

    def _closed_form(self):
        factor = (1.-min(self.total_iters,self.last_epoch)/self.total_iters) ** self.power
        return [rate * factor for rate in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _number(self.gamma, 'gamma')

    def get_lr(self):
        rates = [group['lr'] for group in self.optimizer.param_groups]
        return rates if self._is_initial else [rate * self.gamma for rate in rates]

    def _closed_form(self):
        return [rate * self.gamma ** self.last_epoch for rate in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _integer(self.T_max, 'T_max', 1)
        _number(self.eta_min, 'eta_min')

    def get_lr(self):
        rates = [group['lr'] for group in self.optimizer.param_groups]
        if self._is_initial:
            return rates
        if self._step_count == 1 and self.last_epoch > 0:
            return self._closed_form()
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [rate + (base - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                    for rate, base in zip(rates, self.base_lrs)]
        factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
        return [factor * (rate - self.eta_min) + self.eta_min for rate in rates]

    def _closed_form(self):
        return [self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for base in self.base_lrs]


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine cycles with integer automatic or fractional explicit epochs."""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0., last_epoch=-1):
        self.T_0, self.T_mult, self.eta_min = T_0, T_mult, eta_min
        self.T_i, self.T_cur = T_0, last_epoch
        self._validate_options()
        super().__init__(optimizer, last_epoch)

    def _validate_options(self):
        _integer(self.T_0, 'T_0', 1)
        _integer(self.T_mult, 'T_mult', 1)
        _number(self.eta_min, 'eta_min')
        _integer(self.T_i, 'T_i', 1)
        cycle = self.T_i
        if self.T_mult > 1:
            while cycle > self.T_0 and cycle % self.T_mult == 0:
                cycle //= self.T_mult
        if cycle != self.T_0:
            raise ValueError('T_i must be a valid restart cycle length')
        if not getattr(self, '_is_initial', True):
            _number(self.T_cur, 'T_cur')
            if self.T_cur >= self.T_i:
                raise ValueError('T_cur must be inside the current restart cycle')

    def get_lr(self):
        factor = (1. + math.cos(math.pi * self.T_cur / self.T_i)) / 2.
        return [self.eta_min + (base - self.eta_min) * factor for base in self.base_lrs]

    def step(self, epoch=None):
        self._validate()
        candidate = copy.copy(self)
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
            candidate.T_cur += 1
            if candidate.T_cur >= candidate.T_i:
                candidate.T_cur %= candidate.T_i
                candidate.T_i *= candidate.T_mult
        else:
            _number(epoch, 'epoch')
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    candidate.T_cur = epoch % self.T_0
                else:
                    cycle = int(math.log(epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult))
                    candidate.T_cur = epoch - self.T_0 * (self.T_mult ** cycle - 1) / (self.T_mult - 1)
                    candidate.T_i = self.T_0 * self.T_mult ** cycle
            else:
                candidate.T_i, candidate.T_cur = self.T_0, epoch
        candidate.last_epoch = math.floor(epoch)
        candidate._is_initial = False
        candidate._validate()
        rates = candidate.get_lr()
        self._apply(rates)
        self.T_i, self.T_cur = candidate.T_i, candidate.T_cur
        self.last_epoch, self._last_lr = candidate.last_epoch, list(rates)

    def load_state_dict(self, state_dict):
        if type(state_dict) is not dict or state_dict.get('_is_initial') is not False:
            raise ValueError('invalid warm-restart scheduler initialization state')
        super().load_state_dict(state_dict)


class ReduceLROnPlateau(_State):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self._optimizer(optimizer)
        self.mode, self.factor, self.patience = mode, factor, patience
        self.threshold, self.threshold_mode, self.cooldown, self.eps = threshold, threshold_mode, cooldown, eps
        self.default_min_lr = None if isinstance(min_lr, (list, tuple)) else min_lr
        self.min_lrs = list(min_lr) if self.default_min_lr is None else [min_lr] * len(optimizer.param_groups)
        self.last_epoch, self.cooldown_counter, self.num_bad_epochs = 0, 0, 0
        self.mode_worse = math.inf if mode == 'min' else -math.inf
        self.best = self.mode_worse
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
        self._validate()

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _validate(self):
        self.optimizer._validate_membership()
        if self.mode not in ('min', 'max') or self.threshold_mode not in ('rel', 'abs'):
            raise ValueError('invalid plateau mode or threshold_mode')
        if _number(self.factor, 'factor') >= 1:
            raise ValueError('factor must be below one')
        _number(self.threshold, 'threshold')
        _number(self.eps, 'eps')
        for key in ('patience', 'cooldown', 'last_epoch', 'cooldown_counter', 'num_bad_epochs'):
            _integer(getattr(self, key), key)
        if self.cooldown_counter > self.cooldown:
            raise ValueError('cooldown counter exceeds cooldown')
        self._rates(self.min_lrs, 'min_lrs')
        self._rates(self._last_lr, '_last_lr')
        if self.default_min_lr is not None:
            _number(self.default_min_lr, 'default_min_lr')
        if type(self.best) not in (int, float) or math.isnan(self.best) or self.mode_worse != (math.inf if self.mode == 'min' else -math.inf):
            raise ValueError('invalid plateau best/mode_worse')

    def _better(self, value):
        if self.mode == 'min':
            return value < (self.best * (1 - self.threshold) if self.threshold_mode == 'rel' else self.best - self.threshold)
        return value > (self.best * (1 + self.threshold) if self.threshold_mode == 'rel' else self.best + self.threshold)

    def step(self, metrics, epoch=None):
        self._validate()
        metric = float(metrics.item() if isinstance(metrics, Tensor) else metrics)
        if epoch is not None:
            _integer(epoch, 'epoch')
        candidate = copy.copy(self)
        candidate.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        if candidate._better(metric):
            candidate.best, candidate.num_bad_epochs = metric, 0
        else:
            candidate.num_bad_epochs += 1
        if candidate.in_cooldown:
            candidate.cooldown_counter -= 1
            candidate.num_bad_epochs = 0
        rates = [group['lr'] for group in self.optimizer.param_groups]
        if candidate.num_bad_epochs > candidate.patience:
            rates = [max(rate * candidate.factor, minimum) if rate - max(rate * candidate.factor, minimum) > candidate.eps else rate
                     for rate, minimum in zip(rates, candidate.min_lrs)]
            candidate.cooldown_counter, candidate.num_bad_epochs = candidate.cooldown, 0
        self._apply(rates)
        self.__dict__.update(candidate.state_dict())
        self._last_lr = list(rates)


def _copy_scheduler(scheduler, memo=None):
    memo = {} if memo is None else memo
    if id(scheduler) in memo:
        return memo[id(scheduler)]
    result = copy.copy(scheduler)
    memo[id(scheduler)] = result
    result.__dict__ = {}
    for name, value in scheduler.__dict__.items():
        if name == 'optimizer':
            result.optimizer = value
        elif name == '_schedulers':
            result._schedulers = [_copy_scheduler(child, memo) for child in value]
        else:
            result.__dict__[name] = copy.deepcopy(value)
    return result


def _commit_scheduler(target, source, visited=None):
    visited = set() if visited is None else visited
    if id(target) in visited:
        return
    visited.add(id(target))
    if hasattr(target, '_schedulers'):
        for left, right in zip(target._schedulers, source._schedulers):
            _commit_scheduler(left, right, visited)
    if hasattr(target, 'lr_lambdas'):
        for left, right in zip(target.lr_lambdas, source.lr_lambdas):
            if not isinstance(left, types.FunctionType):
                left.__dict__.clear()
                left.__dict__.update(right.__dict__)
    if hasattr(target, '_scale_fn_custom'):
        left, right = target._scale_fn_custom, source._scale_fn_custom
        if left is not None and not isinstance(left, types.FunctionType):
            left.__dict__.clear()
            left.__dict__.update(right.__dict__)
    for name, value in source.__dict__.items():
        if name not in ('optimizer', '_schedulers', 'lr_lambdas', '_scale_fn_custom'):
            target.__dict__[name] = value


class ChainedScheduler(LRScheduler):
    """Step compatible child schedules in order, sharing one optimizer.

    State loads validate every child before committing. Steps are sequential:
    a later callback failure can leave earlier schedules already advanced.
    Callback effects on external objects are not rolled back.
    """
    def __init__(self, schedulers, optimizer=None):
        if not isinstance(schedulers, (list, tuple)) or not schedulers:
            raise ValueError('at least one scheduler is required')
        if any(not isinstance(child, LRScheduler) for child in schedulers):
            raise TypeError('children must be learning-rate schedules without metrics')
        self._schedulers = list(schedulers)
        self._optimizer(self._schedulers[0].optimizer if optimizer is None else optimizer)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._validate()

    def _validate(self):
        self.optimizer._validate_membership()
        self._rates(self._last_lr, '_last_lr')
        if not isinstance(self._schedulers, list) or not self._schedulers:
            raise ValueError('at least one scheduler is required')
        for child in self._schedulers:
            if not isinstance(child, LRScheduler) or isinstance(child, ReduceLROnPlateau):
                raise TypeError('children must be learning-rate schedules without metrics')
            if child.optimizer is not self.optimizer:
                raise ValueError('all schedules must share one optimizer')
            child._validate()

    def step(self):
        self._validate()
        for child in self._schedulers:
            child.step()
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        values = {name: copy.deepcopy(value) for name, value in self.__dict__.items()
                  if name not in ('optimizer', '_schedulers')}
        values['_schedulers'] = [child.state_dict() for child in self._schedulers]
        return values

    def load_state_dict(self, state_dict):
        if type(state_dict) is not dict or set(state_dict) != set(self.state_dict()):
            raise ValueError('composite scheduler state schema mismatch')
        children = state_dict['_schedulers']
        if type(children) is not list or len(children) != len(self._schedulers):
            raise ValueError('saved scheduler count mismatch')
        candidate = _copy_scheduler(self)
        candidate.__dict__.update(copy.deepcopy({key: value for key, value in state_dict.items() if key != '_schedulers'}))
        for child, saved in zip(candidate._schedulers, children):
            child.load_state_dict(saved)
        candidate._validate()
        _commit_scheduler(self, candidate)


class SequentialLR(ChainedScheduler):
    """Switch schedules at nondecreasing integer milestones.

    Nested composite initialization and duplicate child objects are not supported here.
    Saved schedules must be reconstructed with the same child structure.
    """
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        _integer(last_epoch, 'last_epoch', -1)
        if not isinstance(milestones, (list, tuple)):
            raise TypeError('milestones must be a sequence of integers')
        self._milestones = list(milestones)
        self.last_epoch = last_epoch + 1
        super().__init__(schedulers, optimizer)
        for child in self._schedulers:
            if isinstance(child, ChainedScheduler):
                raise NotImplementedError('SequentialLR does not support composite children')
        if len({id(child) for child in self._schedulers}) != len(self._schedulers):
            raise NotImplementedError('SequentialLR requires distinct child scheduler objects')
        previous = _copy_scheduler(self)
        rates = [group['lr'] for group in optimizer.param_groups]
        configurations = optimizer._native.configurations()
        try:
            self._apply([group['initial_lr'] for group in optimizer.param_groups])
            for child in self._schedulers:
                child.last_epoch -= 1
            index = bisect_right(self._milestones, 0)
            child = self._schedulers[index]
            child._step_count, child._is_initial = 0, True
            try:
                child.step()
            finally:
                child._is_initial = False
            self._last_lr = child.get_last_lr()
        except BaseException:
            optimizer._native.set_configurations(configurations)
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
            _commit_scheduler(self, previous)
            raise

    def _validate(self):
        super()._validate()
        _integer(self.last_epoch, 'last_epoch')
        if type(self._milestones) is not list or len(self._milestones) != len(self._schedulers) - 1:
            raise ValueError('milestones must separate adjacent schedules')
        for value in self._milestones:
            _integer(value, 'milestone')
        if self._milestones != sorted(self._milestones):
            raise ValueError('milestones must be nondecreasing')

    def step(self):
        self._validate()
        epoch = self.last_epoch + 1
        index = bisect_right(self._milestones, epoch)
        child = self._schedulers[index]
        if index > 0 and self._milestones[index - 1] == epoch:
            candidate = copy.copy(child)
            candidate.last_epoch = 0
            rates = candidate._closed_form() if hasattr(candidate, '_closed_form') else candidate.get_lr()
            child._apply(rates)
            child.last_epoch, child._last_lr = 0, list(rates)
        else:
            child.step()
        self.last_epoch, self._last_lr = epoch, child.get_last_lr()


class CyclicLR(LRScheduler):
    """Batchwise scalar LR and momentum cycles with owning callable state.

    Reconstruct a compatible scale_fn before loading; executable functions and
    closure environments are not serialized. External callback effects are not
    rolled back. RMSprop momentum cannot cross zero after state initialization.
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                 mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle',
                 cycle_momentum=True, base_momentum=.8, max_momentum=.9, last_epoch=-1):
        self._optimizer(optimizer)
        _integer(last_epoch,'last_epoch',-1)
        count=len(optimizer.param_groups)
        expand=lambda value:list(value) if isinstance(value,(list,tuple)) else [value]*count
        base_lrs,self.max_lrs=expand(base_lr),expand(max_lr)
        self._rates(base_lrs,'base_lr');self._rates(self.max_lrs,'max_lr')
        up=_number(step_size_up,'step_size_up',positive=True)
        down=up if step_size_down is None else _number(step_size_down,'step_size_down',positive=True)
        self.total_size,self.step_ratio=up+down,up/(up+down)
        self.mode,self.gamma,self._scale_fn_custom,self.scale_mode=mode,gamma,scale_fn,scale_mode
        if scale_fn is None:
            self.scale_mode='iterations' if mode=='exp_range' else 'cycle'
        elif not callable(scale_fn) or (not isinstance(scale_fn,types.FunctionType) and type(getattr(scale_fn,'__dict__',None)) is not dict):
            raise TypeError('scale_fn must be a function or callable with ordinary attributes')
        self.cycle_momentum=cycle_momentum
        if cycle_momentum:
            if 'betas' not in optimizer.defaults and 'momentum' not in optimizer.defaults:
                raise ValueError('cycle_momentum requires optimizer momentum or beta1')
            self.use_beta1='betas' in optimizer.defaults
            self.base_momentums,self.max_momentums=expand(base_momentum),expand(max_momentum)
        self._validate_options()
        previous=[dict(group) for group in optimizer.param_groups]
        configurations=optimizer._native.configurations()
        try:
            if last_epoch == -1:
                for i,group in enumerate(optimizer.param_groups):
                    group['lr']=base_lrs[i]
                    if cycle_momentum:
                        if self.use_beta1:group['betas']=(self.max_momentums[i],*group['betas'][1:])
                        else:group['momentum']=self.max_momentums[i]
                        group['max_momentum'],group['base_momentum']=self.max_momentums[i],self.base_momentums[i]
            super().__init__(optimizer,last_epoch)
            self.base_lrs=base_lrs
        except BaseException:
            optimizer._native.set_configurations(configurations)
            for group,old in zip(optimizer.param_groups,previous):group.clear();group.update(old)
            raise

    def _validate_options(self):
        _number(self.total_size,'total_size',positive=True)
        if type(self.step_ratio) not in (int,float) or not 0 < self.step_ratio < 1:
            raise ValueError('step_ratio must be between zero and one')
        if self._scale_fn_custom is None and self.mode not in ('triangular','triangular2','exp_range'):
            raise ValueError('invalid cyclic mode')
        if self.scale_mode not in ('cycle','iterations'):
            raise ValueError('invalid scale_mode')
        _number(self.gamma,'gamma')
        self._rates(self.max_lrs,'max_lrs')
        if type(self.cycle_momentum) is not bool:
            raise TypeError('cycle_momentum must be bool')
        if self.cycle_momentum:
            if not all(hasattr(self,name) for name in ('use_beta1','base_momentums','max_momentums')):
                raise ValueError('missing cyclic momentum configuration')
            if type(self.use_beta1) is not bool or self.use_beta1 != ('betas' in self.optimizer.defaults):
                raise ValueError('momentum/beta1 binding mismatch')
            for name in ('base_momentums','max_momentums'):
                self._rates(getattr(self,name),name)
                if self.use_beta1 and any(value >= 1 or struct.unpack('<f',struct.pack('<f',value))[0] >= 1 for value in getattr(self,name)):
                    raise ValueError('beta1 momentum must remain below one in float32')

    def scale_fn(self,x):
        if self._scale_fn_custom is not None:return self._scale_fn_custom(x)
        if self.mode=='triangular':return 1.
        if self.mode=='triangular2':return 1./2.**(x-1)
        return self.gamma**x

    def _values(self):
        cycle=math.floor(1+self.last_epoch/self.total_size)
        x=1.+self.last_epoch/self.total_size-cycle
        height=x/self.step_ratio if x <= self.step_ratio else (x-1)/(self.step_ratio-1)
        argument=cycle if self.scale_mode=='cycle' else self.last_epoch
        rates=[base+(maximum-base)*height*self.scale_fn(argument) for base,maximum in zip(self.base_lrs,self.max_lrs)]
        momentums=None
        if self.cycle_momentum:
            momentums=[maximum-(maximum-base)*height*self.scale_fn(argument) for base,maximum in zip(self.base_momentums,self.max_momentums)]
        self._rates(rates,'learning rates')
        return rates,momentums

    def _apply_cycle(self,rates,momentums):
        self.optimizer._validate_membership()
        groups=[dict(group) for group in self.optimizer.param_groups]
        if rates is not None:
            self._rates(rates,'learning rates')
            for group,rate in zip(groups,rates):group['lr']=rate
        if momentums is not None:
            self._rates(momentums,'momentums')
            for group,momentum in zip(groups,momentums):
                if self.use_beta1:group['betas']=(momentum,*group['betas'][1:])
                else:group['momentum']=momentum
        self.optimizer._native.set_configurations(self.optimizer._validated_configurations(groups))
        for group,new in zip(self.optimizer.param_groups,groups):
            group['lr']=new['lr']
            if momentums is not None:
                key='betas' if self.use_beta1 else 'momentum'
                group[key]=new[key]

    def get_lr(self):
        self._validate()
        rates,momentums=self._values()
        if momentums is not None:self._apply_cycle(None,momentums)
        return rates

    def step(self,epoch=None):
        self._validate()
        if epoch is not None:_integer(epoch,'epoch')
        candidate=copy.copy(self)
        candidate._step_count+=1
        candidate.last_epoch=self.last_epoch+1 if epoch is None else epoch
        rates,momentums=candidate._values()
        self._apply_cycle(rates,momentums)
        self.last_epoch,self._step_count,self._last_lr=candidate.last_epoch,candidate._step_count,list(rates)

    def state_dict(self):
        values=copy.deepcopy({key:value for key,value in self.__dict__.items() if key not in ('optimizer','_scale_fn_custom')})
        callback=self._scale_fn_custom
        values['_scale_fn_custom']=None if callback is None or isinstance(callback,types.FunctionType) else copy.deepcopy(callback.__dict__)
        return values

    def load_state_dict(self,state_dict):
        if type(state_dict) is not dict or set(state_dict) != set(self.state_dict()):
            raise ValueError('cyclic scheduler state schema mismatch')
        attributes=state_dict['_scale_fn_custom'];callback=self._scale_fn_custom
        simple=callback is None or isinstance(callback,types.FunctionType)
        if (simple and attributes is not None) or (not simple and type(attributes) is not dict):
            raise ValueError('scale function state does not match the reconstructed callback')
        candidate=copy.copy(self)
        candidate.__dict__.update(copy.deepcopy({key:value for key,value in state_dict.items() if key!='_scale_fn_custom'}))
        owned=None if simple else copy.deepcopy(attributes)
        candidate._validate()
        self.__dict__.update({key:value for key,value in candidate.__dict__.items() if key not in ('optimizer','_scale_fn_custom')})
        if not simple:callback.__dict__.clear();callback.__dict__.update(owned)


class OneCycleLR(LRScheduler):
    """Batchwise one-cycle LR and momentum schedules over native groups.

    Negative rates from linear extrapolation after the training cycle reject
    before changing rates, momenta, or counters. Tensor-valued settings and
    zero-width evaluated phases are unsupported. As with CyclicLR, RMSprop
    momentum cannot cross zero after state initialization.
    """
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
                 pct_start=.3, anneal_strategy='cos', cycle_momentum=True,
                 base_momentum=.85, max_momentum=.95, div_factor=25., final_div_factor=1e4,
                 three_phase=False, last_epoch=-1):
        self._optimizer(optimizer)
        _integer(last_epoch,'last_epoch',-1)
        if total_steps is None:
            _integer(epochs,'epochs',1);_integer(steps_per_epoch,'steps_per_epoch',1)
            total_steps=epochs*steps_per_epoch
        self.total_steps=_integer(total_steps,'total_steps',1)
        if type(pct_start) is not float or not math.isfinite(pct_start) or not 0 <= pct_start <= 1:
            raise ValueError('pct_start must be a finite float between zero and one')
        if type(three_phase) is not bool or type(cycle_momentum) is not bool:
            raise TypeError('three_phase and cycle_momentum must be bool')
        if anneal_strategy not in ('cos','linear'):
            raise ValueError('anneal_strategy must be cos or linear')
        self._anneal_func_type,self.cycle_momentum=anneal_strategy,cycle_momentum
        div_factor=_number(div_factor,'div_factor',positive=True)
        final_div_factor=_number(final_div_factor,'final_div_factor',positive=True)
        phase=lambda end,start_lr,end_lr,start_momentum,end_momentum:dict(
            end_step=end,start_lr=start_lr,end_lr=end_lr,start_momentum=start_momentum,end_momentum=end_momentum)
        self._schedule_phases=[phase(float(pct_start*total_steps)-1,'initial_lr','max_lr','max_momentum','base_momentum')]
        if three_phase:
            self._schedule_phases.extend([
                phase(float(2*pct_start*total_steps)-2,'max_lr','initial_lr','base_momentum','max_momentum'),
                phase(total_steps-1,'initial_lr','min_lr','max_momentum','max_momentum')])
        else:
            self._schedule_phases.append(phase(total_steps-1,'max_lr','min_lr','base_momentum','max_momentum'))
        count=len(optimizer.param_groups)
        expand=lambda value:list(value) if isinstance(value,(tuple,list)) else [value]*count
        maximums=expand(max_lr)
        self._rates(maximums,'max_lr')
        if cycle_momentum:
            if 'betas' not in optimizer.defaults and 'momentum' not in optimizer.defaults:
                raise ValueError('cycle_momentum requires optimizer momentum or beta1')
            self.use_beta1='betas' in optimizer.defaults
            lows,highs=expand(base_momentum),expand(max_momentum)
            self._rates(lows,'base_momentum');self._rates(highs,'max_momentum')
        previous=[dict(group) for group in optimizer.param_groups]
        configurations=optimizer._native.configurations()
        try:
            if last_epoch == -1:
                for index,group in enumerate(optimizer.param_groups):
                    group['max_lr']=maximums[index]
                    group['initial_lr']=maximums[index]/div_factor
                    group['min_lr']=group['initial_lr']/final_div_factor
                    if cycle_momentum:
                        group['base_momentum'],group['max_momentum']=lows[index],highs[index]
                        if self.use_beta1:group['betas']=(highs[index],*group['betas'][1:])
                        else:group['momentum']=highs[index]
            self._validate_options()
            super().__init__(optimizer,last_epoch)
        except BaseException:
            optimizer._native.set_configurations(configurations)
            for group,old in zip(optimizer.param_groups,previous):group.clear();group.update(old)
            raise

    def _validate_options(self):
        _integer(self.total_steps,'total_steps',1)
        if self._anneal_func_type not in ('cos','linear'):
            raise ValueError('invalid one-cycle anneal strategy')
        if type(self.cycle_momentum) is not bool:
            raise TypeError('cycle_momentum must be bool')
        if type(self._schedule_phases) is not list or len(self._schedule_phases) not in (2,3):
            raise ValueError('one-cycle requires two or three phases')
        triples=[('initial_lr','max_lr','max_momentum','base_momentum')]
        triples += ([('max_lr','initial_lr','base_momentum','max_momentum'),
                     ('initial_lr','min_lr','max_momentum','max_momentum')]
                    if len(self._schedule_phases)==3 else [('max_lr','min_lr','base_momentum','max_momentum')])
        for phase,values in zip(self._schedule_phases,triples):
            if type(phase) is not dict or set(phase) != {'end_step','start_lr','end_lr','start_momentum','end_momentum'}:
                raise ValueError('invalid one-cycle phase schema')
            if tuple(phase[key] for key in ('start_lr','end_lr','start_momentum','end_momentum')) != values:
                raise ValueError('invalid one-cycle phase bindings')
            if type(phase['end_step']) not in (int,float) or not math.isfinite(phase['end_step']):
                raise ValueError('invalid one-cycle phase endpoint')
        first=self._schedule_phases[0]['end_step']
        if not -1 <= first <= self.total_steps-1 or self._schedule_phases[-1]['end_step'] != self.total_steps-1:
            raise ValueError('one-cycle phase endpoints disagree with total_steps')
        if len(self._schedule_phases)==3 and not math.isclose(self._schedule_phases[1]['end_step'],2*first,rel_tol=1e-14,abs_tol=1e-14):
            raise ValueError('one-cycle first two phases must be symmetric')
        if self.cycle_momentum:
            if type(getattr(self,'use_beta1',None)) is not bool or self.use_beta1 != ('betas' in self.optimizer.defaults):
                raise ValueError('momentum/beta1 binding mismatch')
            if not self.use_beta1 and 'momentum' not in self.optimizer.defaults:
                raise ValueError('optimizer does not support momentum')
        for group in self.optimizer.param_groups:
            self.optimizer._validate_group(group)
            for key in ('initial_lr','max_lr','min_lr')+(('base_momentum','max_momentum') if self.cycle_momentum else ()):
                if key not in group:raise ValueError('missing optimizer one-cycle metadata: '+key)
                _number(group[key],key)
                if self.cycle_momentum and self.use_beta1 and key.endswith('momentum') and struct.unpack('<f',struct.pack('<f',group[key]))[0] >= 1:
                    raise ValueError('beta1 momentum must remain below one in float32')

    def _anneal_func(self,start,end,pct):
        if self._anneal_func_type=='cos':return end+(start-end)/2.*(math.cos(math.pi*pct)+1)
        return (end-start)*pct+start

    def _validate(self):
        super()._validate()
        if self.last_epoch > self.total_steps:
            raise ValueError('one-cycle schedule exceeded total_steps')

    def _values(self):
        if self.last_epoch > self.total_steps:
            raise ValueError('one-cycle schedule exceeded total_steps')
        rates,momentums=[],[] if self.cycle_momentum else None
        for group in self.optimizer.param_groups:
            start_step=0.
            for index,phase in enumerate(self._schedule_phases):
                end_step=phase['end_step']
                if self.last_epoch <= end_step or index==len(self._schedule_phases)-1:
                    if end_step==start_step:raise ValueError('one-cycle evaluated phase has zero width')
                    pct=(self.last_epoch-start_step)/(end_step-start_step)
                    rates.append(self._anneal_func(group[phase['start_lr']],group[phase['end_lr']],pct))
                    if self.cycle_momentum:
                        momentums.append(self._anneal_func(group[phase['start_momentum']],group[phase['end_momentum']],pct))
                    break
                start_step=end_step
        self._rates(rates,'one-cycle learning rates (negative extrapolation is unsupported)')
        return rates,momentums

    _apply_cycle=CyclicLR._apply_cycle
    get_lr=CyclicLR.get_lr
    step=CyclicLR.step
