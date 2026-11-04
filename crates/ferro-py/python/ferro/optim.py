"""Native optimizers with ordered parameter groups and owning state dictionaries.

Elementwise optimizers skip missing/frozen gradients; LBFGS documents its
coupled-history exception. Python steps retain gradient handles until zero_grad,
matching PyTorch; raw Rust optimizer steps still consume gradients. Python
optimizer steps require eager execution and do not support inference capture.
Active manual gradients that share any parameter's storage are rejected before
updates; shared parameter leaves with independently computed gradients work.
Membership changes use add_param_group. State dictionaries use Ferro tensors
and positional IDs, not Torch serialization files. Loading is atomic on CPU.
SGD, Adam, AdamW, RMSprop and Adagrad also accept CPU float64 parameters
and exact float64 scalar settings. Other optimizers require float32. Plain inputs must be whole contiguous CPU
floating leaves; their identity,
gradient slot and requires_grad flag are preserved. Immutable structural zeros
must be cloned first; retained nonleaf tensors are unsupported. Native training
checkpoints still require optimizer slots bound to registered model Parameters;
external plain leaves use explicit state_dict I/O.
"""
import math
import struct
from ._native import _Optimizer, Tensor, _is_capturing
from .nn import Parameter
from ._optimizer_hooks import OptimizerHooks

_PAIR_OPTIONS = ('betas', 'etas', 'step_sizes')
_BOOL_OPTIONS = ('nesterov', 'centered', 'maximize', 'decoupled_weight_decay', 'amsgrad')
_SCALAR_STATE = ('eta', 'mu', 'mu_product')
_NONE_FLOAT_BITS = 0x7ff8000000000000
_GROUP_METADATA = ('initial_lr', 'max_lr', 'min_lr', 'max_momentum', 'base_momentum')


def _parameter_tensor(parameter):
    return parameter.tensor() if isinstance(parameter, Parameter) else parameter


def _zero_parameter_gradients(parameters, set_to_none):
    if type(set_to_none) is not bool:
        raise TypeError('set_to_none must be bool')
    for parameter in parameters:
        if set_to_none:
            parameter.zero_grad()
        else:
            gradient = _parameter_tensor(parameter).grad
            if gradient is not None:
                gradient.copy_((Tensor.from_f64([0.] * math.prod(gradient.shape), gradient.shape) if gradient.dtype == 'float64' else Tensor.zeros(gradient.shape)).to(gradient.device))


def _number(value, name, *, positive=False, float64=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    if float64:
        return float(value)
    try:
        native = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError(f'{name} is outside float32 range') from None
    if not math.isfinite(native) or (positive and native == 0):
        raise ValueError(f'{name} is outside supported float32 range')
    return float(value)


def _sgd_precise(group):
    return any(_parameter_tensor(p).dtype == 'float64' for p in group['params'])


def _sgd_decode(config):
    if len(config) == 25:
        values = [struct.unpack('<d', struct.pack('<Q', sum(int(chunk) << (16*j) for j, chunk in enumerate(config[i:i+4]))))[0]
                  for i in range(5, 25, 4)]
        return dict(lr=values[0], momentum=values[1], weight_decay=values[2], dampening=values[3],
                    nesterov=bool(config[2]), maximize=bool(config[3]))
    return dict(lr=config[1], momentum=config[2], nesterov=bool(config[3]),
                weight_decay=config[5] if len(config) == 8 else 0., dampening=config[6] if len(config) == 8 else 0.,
                maximize=bool(config[7]) if len(config) == 8 else False)


def _adam_decode(config):
    if len(config) == 29:
        values = [struct.unpack('<d', struct.pack('<Q', sum(int(chunk) << (16*j) for j, chunk in enumerate(config[i:i+4]))))[0]
                  for i in range(5, 29, 4)]
        return dict(lr=values[0], betas=tuple(values[1:3]), eps=values[3], weight_decay=values[4],
                    amsgrad=bool(config[2]), maximize=bool(config[3]))
    return dict(lr=config[1], betas=tuple(config[2:4]), eps=config[4],
                weight_decay=config[5] if config[0] == 3. else config[6] if len(config) == 9 else 0.,
                amsgrad=bool(config[7]) if len(config) == 9 else False,
                maximize=bool(config[8]) if len(config) == 9 else False)


def _adaptive_decode(config):
    """The exact scalar schema is restricted to RMSprop and Adagrad."""
    kind = int(config[0])
    names = ('lr', 'alpha', 'eps', 'weight_decay', 'momentum') if kind == 5 else ('lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps')
    if len(config) == 25:
        values = [struct.unpack('<d', struct.pack('<Q', sum(int(chunk) << (16*j) for j, chunk in enumerate(config[i:i+4]))))[0]
                  for i in range(5, 25, 4)]
        result = dict(zip(names, values), maximize=bool(config[3]))
        if kind == 5:
            result['centered'] = bool(config[2])
        return result
    result = dict(zip(names, config[1:6]), maximize=bool(config[-1]))
    if kind == 5:
        result['centered'] = bool(config[6])
    return result


class _Base(OptimizerHooks):
    _adaptive = False
    _pair_options = _PAIR_OPTIONS

    def _initialize(self, params, defaults):
        if isinstance(params, Tensor):
            raise TypeError('optimizer params must be an iterable of tensors or parameter groups')
        self.defaults = dict(defaults)
        self.param_groups, self._slots, self._seen = [], [], []
        supplied = list(params)
        if not supplied:
            raise ValueError('optimizer received an empty parameter list')
        groups = supplied if isinstance(supplied[0], dict) else [{'params': supplied}]
        native = []
        for supplied_group in groups:
            group = self._make_group(supplied_group)
            previous = {p for existing in self._slots for p in existing}
            if any(p in previous for p in group['params']):
                raise ValueError('a parameter cannot occur in multiple groups')
            native.append(self._new_native(group))
            self.param_groups.append(group)
            self._slots.append(tuple(group['params']))
            self._seen.append([type(self) is Adagrad] * len(group['params']))
        self._native = native[0] if len(native) == 1 else _Optimizer.grouped(native)
        self.params = tuple(p for group in self._slots for p in group)

    def _make_group(self, supplied):
        if type(supplied) is not dict or 'params' not in supplied:
            raise TypeError('each parameter group must be a dict containing params')
        group = {**self.defaults, **supplied}
        params = supplied['params']
        params = [params] if isinstance(params, Tensor) else list(params)
        if any(not isinstance(p, Tensor) for p in params):
            raise TypeError('optimizers require Parameter objects or plain Tensor leaves')
        if any(not p.is_leaf for p in params):
            raise ValueError('optimizer tensors must be leaves without autograd history')
        if any((p.dtype != 'float32' and not (type(self) in (SGD, Adam, AdamW, RMSprop, Adagrad) and p.dtype == 'float64' and p.device == 'cpu')) or (not isinstance(p, Parameter) and p.device != 'cpu') for p in params):
            raise ValueError('optimizers require float32 parameters; SGD/Adam/AdamW/RMSprop/Adagrad additionally support CPU float64; plain leaves require CPU storage')
        group['params'] = list(dict.fromkeys(params))
        self._validate_group(group)
        return group

    def _validate_group(self, group):
        allowed = set(self.defaults) | {'params', *_GROUP_METADATA}
        if set(group) - allowed or not set(self.defaults) <= set(group):
            raise ValueError('unsupported or missing optimizer group option')
        precise = type(self) in (SGD, Adam, AdamW, RMSprop, Adagrad) and _sgd_precise(group)
        if any(_parameter_tensor(p).dtype != 'float32' and not (precise and _parameter_tensor(p).dtype == 'float64' and _parameter_tensor(p).device == 'cpu') for p in group['params']):
            raise ValueError('this optimizer requires float32 parameters; SGD/Adam/AdamW/RMSprop/Adagrad also support CPU float64')
        if precise and type(self) in (Adam, AdamW, RMSprop, Adagrad) and any(_parameter_tensor(p).device != 'cpu' for p in group['params']):
            raise ValueError('mixed float64 optimizer groups require CPU parameters')
        wide = precise and all(_parameter_tensor(p).dtype == 'float64' for p in group['params'])
        _number(group['lr'], 'lr', float64=wide)
        for key in _GROUP_METADATA:
            if key in group:
                _number(group[key], key, float64=wide)
        if self._adaptive:
            for key in self._options:
                if key in _BOOL_OPTIONS:
                    if type(group[key]) is not bool:
                        raise ValueError(f'{key} must be bool')
                elif key in self._pair_options:
                    values = group[key]
                    if not isinstance(values, (list, tuple)) or len(values) != 2:
                        raise ValueError(f'{key} must be a pair')
                    for value in values:
                        _number(value, key)
                        if key == 'betas' and (value >= 1 or struct.unpack('<f', struct.pack('<f', value))[0] >= 1):
                            raise ValueError('betas must remain below one in float32')
                    rounded = [struct.unpack('<f', struct.pack('<f', value))[0] for value in values]
                    if key == 'etas' and not 0 < rounded[0] < 1 < rounded[1]:
                        raise ValueError('etas must satisfy 0 < eta_minus < 1 < eta_plus in float32')
                    if key == 'step_sizes' and values[0] > values[1]:
                        raise ValueError('step_sizes must have minimum <= maximum')
                else:
                    _number(group[key], key, positive=key == 'momentum' and group[key] != 0, float64=wide)
                    if key in ('rho', 'alpha') and type(self) in (RMSprop, Adadelta) and group[key] > 1 and not (type(self) is RMSprop and wide):
                        raise ValueError(f'{key} must be at most one')
            return
        if type(self) is SGD:
            _number(group['momentum'], 'momentum', positive=group['momentum'] != 0, float64=wide)
            _number(group['weight_decay'], 'weight_decay', float64=wide)
            _number(group['dampening'], 'dampening', positive=group['dampening'] != 0, float64=wide)
            if type(group['nesterov']) is not bool or type(group['maximize']) is not bool or (group['nesterov'] and (group['momentum'] <= 0 or group['dampening'] != 0)):
                raise ValueError('nesterov requires positive momentum, zero dampening and bool flags')
            if group['dampening'] and any(_parameter_tensor(p).device != 'cpu' for p in group['params']):
                raise ValueError('SGD dampening currently requires CPU parameters')
        else:
            betas = group['betas']
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError('betas must be a pair')
            for beta in betas:
                value = _number(beta, 'beta', float64=wide)
                if value >= 1 or (not wide and struct.unpack('<f', struct.pack('<f', value))[0] >= 1):
                    raise ValueError('betas must remain below one in the parameter precision')
            _number(group['eps'], 'eps', positive=not precise, float64=wide)
            if type(self) in (Adam, AdamW):
                _number(group['weight_decay'], 'weight_decay', float64=wide)
            if type(self) in (Adam, AdamW):
                if type(group['amsgrad']) is not bool or type(group['maximize']) is not bool:
                    raise ValueError('amsgrad and maximize must be bool')
                if group['amsgrad'] and any(_parameter_tensor(p).device != 'cpu' for p in group['params']):
                    raise ValueError('AMSGrad currently requires CPU parameters')

    def _config(self, group):
        self._validate_group(group)
        if self._adaptive:
            if type(self) in (RMSprop, Adagrad) and _sgd_precise(group):
                values = [float(self._tag), 64., float(group.get('centered', False)), float(group['maximize']), 0.]
                for key in self._options[:5]:
                    bits = struct.unpack('<Q', struct.pack('<d', group[key]))[0]
                    values.extend(float((bits >> shift) & 65535) for shift in (0, 16, 32, 48))
                return values
            return [float(self._tag)] + [float(value) for key in self._options
                                        for value in (group[key] if key in self._pair_options else (group[key],))]
        if type(self) is SGD:
            if _sgd_precise(group):
                values = [1., 64., float(group['nesterov']), float(group['maximize']), 0.]
                for value in (group['lr'], group['momentum'], group['weight_decay'], group['dampening'], 0.):
                    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
                    values.extend(float((bits >> shift) & 65535) for shift in (0, 16, 32, 48))
                return values
            values = [1., group['lr'], group['momentum'], float(group['nesterov']), -1.]
            return values + ([group['weight_decay'], group['dampening'], float(group['maximize'])]
                             if struct.unpack('<f', struct.pack('<f', group['weight_decay']))[0] or group['dampening'] or group['maximize'] else [])
        if type(self) in (Adam, AdamW) and _sgd_precise(group):
            values = [2. if type(self) is Adam else 3., 64., float(group['amsgrad']), float(group['maximize']), 0.]
            for value in (group['lr'], *group['betas'], group['eps'], group['weight_decay'], 0.):
                bits = struct.unpack('<Q', struct.pack('<d', value))[0]
                values.extend(float((bits >> shift) & 65535) for shift in (0, 16, 32, 48))
            return values
        values = [2. if type(self) is Adam else 3., group['lr'], *group['betas'], group['eps']]
        if type(self) is AdamW:
            values.append(group['weight_decay'])
        values.append(-1.)
        if type(self) is Adam and (struct.unpack('<f', struct.pack('<f', group['weight_decay']))[0] or group['amsgrad'] or group['maximize']):
            values.extend([group['weight_decay'], float(group['amsgrad']), float(group['maximize'])])
        if type(self) is AdamW and (group['amsgrad'] or group['maximize']):
            values.extend([float(group['amsgrad']), float(group['maximize'])])
        return values

    def _new_native(self, group):
        if type(self) is SGD and _sgd_precise(group):
            return _Optimizer.sgd(group['params'], self._config(group))
        if type(self) in (Adam, AdamW) and _sgd_precise(group):
            return _Optimizer.adam(group['params'], self._config(group))
        if self._adaptive:
            config = self._config(group)
            if type(self) is Adagrad:
                initial_group = {**group, 'initial_accumulator_value': self.defaults['initial_accumulator_value']}
                native = _Optimizer.adaptive(group['params'], self._config(initial_group))
                native.set_configurations([config])
                return native
            return _Optimizer.adaptive(group['params'], config)
        native = _Optimizer(type(self).__name__, group['params'], group['lr'],
                            *group.get('betas', (.9, .999)), group.get('eps', 1e-8),
                            group.get('weight_decay', 0.), group.get('momentum', 0.), group.get('nesterov', False))
        native.set_configurations([self._config(group)])
        return native

    def _validate_membership(self):
        if len(self.param_groups) != len(self._slots):
            raise ValueError('use add_param_group to change group membership')
        for group, slots in zip(self.param_groups, self._slots):
            current = group.get('params', ()) if type(group) is dict else ()
            if type(group) is not dict or not isinstance(current, (list, tuple)) or len(current) != len(slots) or any(p is not original for p, original in zip(current, slots)):
                raise ValueError('optimizer parameter membership/order cannot be reassigned')

    def _validated_configurations(self, groups=None):
        self._validate_membership()
        groups = self.param_groups if groups is None else groups
        configurations = [self._config(group) for group in groups]
        if type(self) is RMSprop:
            for config, seen, old in zip(configurations, self._seen, self._native.configurations()):
                if any(seen) and (_adaptive_decode(config)['centered'] != _adaptive_decode(old)['centered'] or bool(_adaptive_decode(config)['momentum']) != bool(_adaptive_decode(old)['momentum'])):
                    raise ValueError('RMSprop centered and momentum presence cannot change after state initialization')
        if type(self) in (SGD, Adam, AdamW):
            for group, seen, old in zip(groups, self._seen, self._native.configurations()):
                if type(self) is SGD and bool(group['dampening']) != bool(_sgd_decode(old)['dampening']):
                    raise ValueError('SGD dampening presence stays fixed; construct or restore the desired optimizer configuration')
                if type(self) in (Adam, AdamW) and any(seen) and group['amsgrad'] != _adam_decode(old)['amsgrad']:
                    raise ValueError('amsgrad cannot change after state initialization')
        return configurations

    def _sync(self):
        self._native.set_configurations(self._validated_configurations())

    def add_param_group(self, param_group):
        self._validate_membership()
        group = self._make_group(param_group)
        identities = {id(p) for p in self.params}
        if any(id(p) in identities for p in group['params']):
            raise ValueError('a parameter cannot occur in multiple groups')
        self._native.add_group(self._new_native(group))
        self.param_groups.append(group)
        self._slots.append(tuple(group['params']))
        self._seen.append([False] * len(group['params']))
        self.params = tuple(p for slots in self._slots for p in slots)

    def zero_grad(self, set_to_none=True):
        _zero_parameter_gradients(self.params, set_to_none)

    def _initialize_adagrad_slots(self):
        if type(self) is not Adagrad:
            return
        pending = []
        for gi, (slots, seen) in enumerate(zip(self._slots, self._seen)):
            for pi, (parameter, initialized) in enumerate(zip(slots, seen)):
                if not initialized and parameter.grad is not None:
                    dtype = _parameter_tensor(parameter).dtype
                    initial = _number(self.defaults['initial_accumulator_value'], 'initial_accumulator_value', float64=dtype == 'float64')
                    shape = _parameter_tensor(parameter).shape
                    value = (Tensor.from_f64 if dtype == 'float64' else Tensor)([initial] * math.prod(shape), shape)
                    pending.append((gi, pi, value.reshape([math.prod(shape)])))
        if pending:
            # Reuse the owning native snapshot/atomic restore. Untouched hidden
            # slots keep their state, even if their dtype cannot represent a
            # newly changed default that applies only to active added slots.
            arrays = dict(self._native.state_tensors())
            for gi, pi, value in pending:
                arrays[self._prefix(gi) + f'sum.{pi}'] = value
            self._native.restore_state(self._packed_config(self.param_groups), list(arrays.items()))
            for gi, pi, _ in pending:
                self._seen[gi][pi] = True

    def step(self, closure=None):
        if _is_capturing():
            raise NotImplementedError('Python optimizer steps require eager execution')
        self._native.validate_gradient_storage()
        self._sync()
        loss = None
        if closure is not None:
            from ._grad_mode import enable_grad
            with enable_grad():
                loss = closure()
            self._native.validate_gradient_storage()
            self._sync()
        self._initialize_adagrad_slots()
        active = [[p.grad is not None and (type(self) is not SGD or group['momentum'] != 0)
                   for p in slots] for group, slots in zip(self.param_groups, self._slots)]
        self._native.step(preserve_grads=True)
        self._seen = [[old or now for old, now in zip(seen, flags)] for seen, flags in zip(self._seen, active)]
        return loss

    def _state_shape(self, parameter, name):
        return [] if self._adaptive and name in _SCALAR_STATE else _parameter_tensor(parameter).shape

    def _prefix(self, group):
        return '' if len(self.param_groups) == 1 else f'g{group}.'

    def _packed_config(self, groups):
        configs = [self._config(g) for g in groups]
        if len(configs) == 1:
            return configs[0]
        return [4., float(len(groups))] + [value for g, cfg in zip(groups, configs)
                                          for value in (float(len(g['params'])), float(len(cfg)), *cfg)]

    def state_dict(self):
        """Owning Torch-shaped state with Ferro tensors and positional IDs."""
        self._sync()
        arrays = dict(self._native.state_tensors())
        state, groups, index = {}, [], 0
        for gi, (group, slots) in enumerate(zip(self.param_groups, self._slots)):
            prefix, ids = self._prefix(gi), []
            for pi, p in enumerate(slots):
                ids.append(index)
                if self._adaptive:
                    step = arrays[prefix + f't.{pi}'].item()
                    initialized = bool(arrays[prefix + f'initialized.{pi}'].item()) if type(self) in (ASGD, Rprop) else bool(step or self._seen[gi][pi])
                    if initialized:
                        state[index] = {'step': Tensor([step], [])}
                        for name in self._state_names(group, p):
                            value = arrays[prefix + f'{name}.{pi}']
                            state[index][name] = value.reshape(self._state_shape(p, name))
                elif type(self) is SGD:
                    if self._seen[gi][pi]:
                        state[index] = {'momentum_buffer': arrays[prefix + f'velocity.{pi}'].reshape(_parameter_tensor(p).shape)}
                else:
                    step = arrays[prefix + f't.{pi}'].item() if type(self) is Adam else arrays[prefix + 't'].tolist()[pi]
                    if _sgd_precise(group) and step > 2**24:
                        raise ValueError('exact Adam counter exceeds the public float32 step bound; use a training checkpoint')
                    if step or self._seen[gi][pi]:
                        state[index] = {'step': Tensor([step], []),
                                        'exp_avg': arrays[prefix + f'm.{pi}'].reshape(_parameter_tensor(p).shape),
                                        'exp_avg_sq': arrays[prefix + f'v.{pi}'].reshape(_parameter_tensor(p).shape)}
                        if type(self) in (Adam, AdamW) and group['amsgrad']:
                            state[index]['max_exp_avg_sq'] = arrays[prefix + f'vmax.{pi}'].reshape(_parameter_tensor(p).shape)
                index += 1
            groups.append({**{key: tuple(value) if key in self._pair_options else value for key, value in group.items() if key != 'params'}, 'params': ids})
        return {'state': state, 'param_groups': groups}

    def load_state_dict(self, state_dict):
        """Validate all groups/buffers before one native CPU transaction.

        Positional IDs map onto current groups, as in Torch. Buffer storage must
        match CPU parameter dtype (float64 is supported by SGD, Adam, AdamW, RMSprop and Adagrad). Model parameters
        and pending gradients are unchanged.
        """
        self._validate_membership()
        if type(state_dict) is not dict or set(state_dict) != {'state', 'param_groups'}:
            raise ValueError('optimizer state needs state and param_groups')
        states, saved = state_dict['state'], state_dict['param_groups']
        if type(states) is not dict or not isinstance(saved, (list, tuple)) or len(saved) != len(self._slots):
            raise ValueError('optimizer state/group count mismatch')
        groups, arrays, seen, all_ids = [], [], [], set()
        for gi, (source, slots) in enumerate(zip(saved, self._slots)):
            if type(source) is not dict or not isinstance(source.get('params'), (list, tuple)) or len(source['params']) != len(slots):
                raise ValueError('optimizer group parameter count mismatch')
            ids = source['params']
            if any(type(i) is not int or i < 0 or i in all_ids for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('optimizer state IDs must be unique nonnegative integers')
            all_ids.update(ids)
            group = {**source, 'params': list(slots)}
            for key, value in self._legacy_defaults().items():
                group.setdefault(key, value)
            for key in self._pair_options:
                if key in group:
                    group[key] = tuple(group[key])
            self._validate_group(group)
            groups.append(group)
            prefix, flags, steps = self._prefix(gi), [], []
            for pi, (identifier, parameter) in enumerate(zip(ids, slots)):
                payload = states.get(identifier, {})
                required = ({'step', *self._state_names(group, parameter)} if self._adaptive else
                            {'momentum_buffer'} if type(self) is SGD else {'step', 'exp_avg', 'exp_avg_sq'})
                if type(self) in (Adam, AdamW) and group['amsgrad']:
                    required.add('max_exp_avg_sq')
                if type(payload) is not dict or (payload and set(payload) != required):
                    raise ValueError('optimizer slot state keys mismatch')
                flags.append(bool(payload))
                if type(self) is SGD:
                    values = [('velocity', payload.get('momentum_buffer'))]
                    if group['dampening'] or _sgd_precise(group):
                        arrays.append((prefix + f'initialized.{pi}', Tensor([float(bool(payload))], [])))
                else:
                    step = payload.get('step', 0.)
                    if isinstance(step, Tensor):
                        if step.dtype != 'float32' or step.device != 'cpu' or step.shape != []:
                            raise ValueError('optimizer step must be a CPU float32 scalar')
                        step = step.item()
                    if type(step) not in (int, float) or not math.isfinite(step) or step < 0 or step != int(step) or step > 2**24:
                        raise ValueError('optimizer step must be an exactly representable nonnegative integer')
                    steps.append(float(step))
                    values = ([(name, payload.get(name)) for name in self._all_state_names] if self._adaptive else
                              [('m', payload.get('exp_avg')), ('v', payload.get('exp_avg_sq'))])
                    if type(self) in (Adam, AdamW) and group['amsgrad']:
                        values.append(('vmax', payload.get('max_exp_avg_sq')))
                    if type(self) is Adam or self._adaptive:
                        arrays.append((prefix + f't.{pi}', (Tensor.from_f64 if type(self) is Adam and _sgd_precise(group) else Tensor)([step], [])))
                    if type(self) in (ASGD, Rprop):
                        arrays.append((prefix + f'initialized.{pi}', Tensor([float(bool(payload))], [])))
                for name, value in values:
                    shape = self._state_shape(parameter, name)
                    dtype = _parameter_tensor(parameter).dtype if type(self) in (SGD, Adam, AdamW, RMSprop, Adagrad) else 'float32'
                    if value is None:
                        initial = self.defaults['initial_accumulator_value'] if type(self) is Adagrad and name == 'sum' else group['lr'] if name in ('eta', 'step_size') else 1. if name in ('mu', 'mu_product') else 0.
                        value = (Tensor.from_f64 if dtype == 'float64' else Tensor)([initial] * math.prod(shape), shape)
                    if not isinstance(value, Tensor) or value.dtype != dtype or value.device != 'cpu' or value.shape != shape:
                        raise ValueError('optimizer buffer shape/dtype/device mismatch')
                    arrays.append((prefix + f'{name}.{pi}', value if name in _SCALAR_STATE else value.reshape([math.prod(shape)])))
            if type(self) is AdamW:
                arrays.append((prefix + 't', (Tensor.from_f64 if _sgd_precise(group) else Tensor)(steps, [len(steps)])))
            seen.append(flags)
        if any(type(key) is not int or key not in all_ids for key in states):
            raise ValueError('optimizer state contains an unknown parameter ID')
        self._native.restore_state(self._packed_config(groups), arrays)
        for target, source in zip(self.param_groups, groups):
            target.clear()
            target.update(source)
        self._seen = seen

    def _checkpoint_metadata(self):
        self._validate_membership()
        values = {}
        for gi, (group, seen) in enumerate(zip(self.param_groups, self._seen)):
            self._validate_group(group)
            for key in self.defaults:
                value = group[key]
                if key in self._pair_options:
                    for i, member in enumerate(value):
                        values[f'g{gi}.{key}.{i}'] = _NONE_FLOAT_BITS if member is None else struct.unpack('<Q', struct.pack('<d', member))[0]
                elif key == 'line_search_fn':
                    values[f'g{gi}.{key}'] = int(value is not None)
                elif key in _BOOL_OPTIONS:
                    values[f'g{gi}.{key}'] = int(value)
                else:
                    values[f'g{gi}.{key}'] = struct.unpack('<Q', struct.pack('<d', value))[0]
            for key in _GROUP_METADATA:
                values[f'g{gi}.{key}.present'] = int(key in group)
                values[f'g{gi}.{key}'] = struct.unpack('<Q', struct.pack('<d', group.get(key, 0.)))[0]
            for pi, flag in enumerate(seen):
                values[f'g{gi}.seen.{pi}'] = int(flag)
        return values

    def _legacy_defaults(self):
        if type(self) is SGD:
            return dict(weight_decay=0., dampening=0., maximize=False)
        if type(self) is Adam:
            return dict(weight_decay=0., amsgrad=False, maximize=False)
        if type(self) is AdamW:
            return dict(amsgrad=False, maximize=False)
        return {}

    def _prepare_checkpoint_metadata(self, metadata, native_configuration):
        metadata = dict(metadata)
        additions = self._legacy_defaults()
        for gi in range(len(self._slots)):
            names = {f'g{gi}.{key}' for key in additions}
            missing = names - set(metadata)
            if missing == names:
                for key, value in additions.items():
                    metadata[f'g{gi}.{key}'] = int(value) if key in _BOOL_OPTIONS else struct.unpack('<Q', struct.pack('<d', value))[0]
        names = {name for gi in range(len(self._slots)) for key in _GROUP_METADATA[1:]
                 for name in (f'g{gi}.{key}.present', f'g{gi}.{key}')}
        if names.isdisjoint(metadata):
            metadata.update({name: 0 for name in names})
        if set(metadata) != set(self._checkpoint_metadata()):
            raise ValueError('optimizer metadata/group schema mismatch')
        groups, all_seen = [], []
        decode = lambda bits: struct.unpack('<d', struct.pack('<Q', bits))[0]
        for gi, slots in enumerate(self._slots):
            group = {'params': list(slots)}
            for key in self.defaults:
                if key in self._pair_options:
                    group[key] = tuple(None if key == 'eps' and i == 0 and metadata[f'g{gi}.{key}.{i}'] == _NONE_FLOAT_BITS
                                       else decode(metadata[f'g{gi}.{key}.{i}']) for i in (0, 1))
                elif key == 'line_search_fn':
                    if metadata[f'g{gi}.{key}'] not in (0, 1):
                        raise ValueError('unsupported saved LBFGS line search')
                    group[key] = None if metadata[f'g{gi}.{key}'] == 0 else 'strong_wolfe'
                elif key in _BOOL_OPTIONS:
                    value = metadata[f'g{gi}.{key}']
                    if value not in (0, 1):
                        raise ValueError('invalid saved optimizer bool flag')
                    group[key] = bool(value)
                else:
                    value = decode(metadata[f'g{gi}.{key}'])
                    if type(self) is LBFGS and key in self._integers:
                        if not math.isfinite(value) or value != int(value):
                            raise ValueError('invalid saved LBFGS integer configuration')
                        value = int(value)
                    group[key] = value
            for key in _GROUP_METADATA:
                present = metadata[f'g{gi}.{key}.present']
                if present not in (0, 1):
                    raise ValueError('invalid scheduler metadata presence flag')
                value = decode(metadata[f'g{gi}.{key}'])
                if present:
                    group[key] = value
                elif value != 0.:
                    raise ValueError('unexpected inactive scheduler metadata value')
            self._validate_group(group)
            seen = [metadata[f'g{gi}.seen.{pi}'] for pi in range(len(slots))]
            if any(value not in (0, 1) for value in seen):
                raise ValueError('invalid optimizer initialized flag')
            groups.append(group)
            all_seen.append([bool(value) for value in seen])
        expected = self._packed_config(groups)
        if native_configuration.dtype != 'float32' or native_configuration.shape != [len(expected)]:
            raise ValueError('optimizer configuration shape/dtype mismatch')
        if struct.pack('<' + 'f' * len(expected), *expected) != struct.pack('<' + 'f' * len(expected), *native_configuration.tolist()):
            raise ValueError('optimizer metadata disagrees with native configuration')
        return groups, all_seen

    def _commit_checkpoint_metadata(self, staged=None):
        if staged is None:
            # Legacy single-group checkpoints have only native f32 settings.
            groups = []
            for current, config in zip(self.param_groups, self._native.configurations()):
                group = {'params': current['params'], 'lr': config[1]}
                if type(self) in (RMSprop, Adagrad):
                    group.update(_adaptive_decode(config))
                elif self._adaptive:
                    offset = 1
                    for key in self._options:
                        if key in self._pair_options:
                            group[key], offset = tuple(config[offset:offset + 2]), offset + 2
                        else:
                            group[key] = bool(config[offset]) if key in _BOOL_OPTIONS else config[offset]
                            offset += 1
                elif type(self) is SGD:
                    group.update(_sgd_decode(config))
                else:
                    group.update(_adam_decode(config))
                groups.append(group)
            staged = groups, [[True] * len(slots) for slots in self._slots]
        groups, self._seen = staged
        for target, source in zip(self.param_groups, groups):
            target.clear()
            target.update(source)


class SGD(_Base):
    """Existing float32 kernels plus eager CPU float64 updates.

    Float64 groups preserve Python scalar bits in native checkpoint state.
    Momentum snapshots are contiguous and loading requires the matching CPU
    parameter dtype; implicit cross-dtype state casting is unsupported.
    Live manual gradients are retained by step and cleared by zero_grad().
    zero_grad(False) retains the existing checked whole-leaf copy boundary.
    """
    def __init__(self, params, lr=0.001, momentum=0.0, dampening=0., weight_decay=0., nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None):
        for name, value in (('foreach', foreach), ('fused', fused)):
            if value is not None and type(value) is not bool:
                raise TypeError(name + ' must be bool or None')
            if value is True:
                raise NotImplementedError('SGD ' + name + '=True is unsupported')
        if type(differentiable) is not bool:
            raise TypeError('differentiable must be bool')
        if differentiable:
            raise NotImplementedError('differentiable SGD is unsupported')
        self._initialize(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, dampening=dampening, maximize=maximize))


class Adam(_Base):
    """Adam with Float32 kernels and eager CPU Float64 moments and scalars."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0., amsgrad=False, *, maximize=False):
        self._initialize(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize))


class AdamW(_Base):
    """AdamW with Float32 kernels and eager CPU Float64 moments and scalars."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, *, maximize=False):
        self._initialize(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize))


class RMSprop(_Base):
    """Dense CPU float32/float64 RMSprop with optional centering and momentum.

    Centering and momentum presence stay fixed after a group initializes state;
    rates, nonzero momentum values and other scalar settings remain mutable.
    Double-only groups retain exact finite scalar settings; legacy float32
    option limits remain for mixed groups. Step counters stop at 2**24.
    """
    _adaptive, _tag = True, 5
    _options = ('lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered', 'maximize')
    _all_state_names = ('square_avg', 'momentum_buffer', 'grad_avg')

    def __init__(self, params, lr=.01, alpha=.99, eps=1e-8, weight_decay=0., momentum=0., centered=False, *, maximize=False):
        self._initialize(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                     momentum=momentum, centered=centered, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return ('square_avg',) + (('momentum_buffer',) if group['momentum'] else ()) + (('grad_avg',) if group['centered'] else ())


class Adagrad(_Base):
    """Dense CPU float32/float64 Adagrad with per-slot decay.

    As in Torch, accumulators use the constructor default even when a group's
    initial_accumulator_value differs. Added groups initialize state from the
    current defaults on first use. Options are finite and nonnegative; typed
    state loads require matching dtypes, and step counters stop at 2**24.
    """
    _adaptive, _tag = True, 6
    _options = ('lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps', 'maximize')
    _all_state_names = ('sum',)

    def __init__(self, params, lr=.01, lr_decay=0., weight_decay=0., initial_accumulator_value=0., eps=1e-10, *, maximize=False):
        self._initialize(params, dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                                     initial_accumulator_value=initial_accumulator_value, eps=eps, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class Adamax(_Base):
    """Dense CPU f32 Adamax with infinity-norm state and coupled weight decay."""
    _adaptive, _tag = True, 7
    _options = ('lr', 'betas', 'eps', 'weight_decay', 'maximize')
    _all_state_names = ('exp_avg', 'exp_inf')

    def __init__(self, params, lr=.002, betas=(.9, .999), eps=1e-8, weight_decay=0., *, maximize=False):
        self._initialize(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class Adadelta(_Base):
    """Dense CPU f32 Adadelta with gradient and update running squares."""
    _adaptive, _tag = True, 8
    _options = ('lr', 'rho', 'eps', 'weight_decay', 'maximize')
    _all_state_names = ('square_avg', 'acc_delta')

    def __init__(self, params, lr=1., rho=.9, eps=1e-6, weight_decay=0., *, maximize=False):
        self._initialize(params, dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class ASGD(_Base):
    """Dense CPU f32 averaged SGD; averaged iterates are available in state_dict."""
    _adaptive, _tag = True, 9
    _options = ('lr', 'lambd', 'alpha', 't0', 'weight_decay', 'maximize')
    _all_state_names = ('eta', 'mu', 'ax')

    def __init__(self, params, lr=.01, lambd=1e-4, alpha=.75, t0=1e6, weight_decay=0., *, maximize=False):
        self._initialize(params, dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class Rprop(_Base):
    """Dense CPU f32 resilient propagation with per-element step sizes."""
    _adaptive, _tag = True, 10
    _options = ('lr', 'etas', 'step_sizes', 'maximize')
    _all_state_names = ('prev', 'step_size')

    def __init__(self, params, lr=.01, etas=(.5, 1.2), step_sizes=(1e-6, 50.), *, maximize=False):
        self._initialize(params, dict(lr=lr, etas=tuple(etas), step_sizes=tuple(step_sizes), maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class RAdam(_Base):
    """Dense CPU f32 rectified Adam with optional decoupled weight decay."""
    _adaptive, _tag = True, 11
    _options = ('lr', 'betas', 'eps', 'weight_decay', 'decoupled_weight_decay', 'maximize')
    _all_state_names = ('exp_avg', 'exp_avg_sq')

    def __init__(self, params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0., *, decoupled_weight_decay=False, maximize=False):
        self._initialize(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                     decoupled_weight_decay=decoupled_weight_decay, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class NAdam(_Base):
    """Dense CPU f32 Nesterov Adam with per-slot momentum products."""
    _adaptive, _tag = True, 12
    _options = ('lr', 'betas', 'eps', 'weight_decay', 'momentum_decay', 'decoupled_weight_decay', 'maximize')
    _all_state_names = ('exp_avg', 'exp_avg_sq', 'mu_product')

    def __init__(self, params, lr=.002, betas=(.9, .999), eps=1e-8, weight_decay=0., momentum_decay=.004, *, decoupled_weight_decay=False, maximize=False):
        self._initialize(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                     momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, maximize=maximize))

    def _state_names(self, group, parameter=None):
        return self._all_state_names


class Adafactor(_Base):
    """Dense CPU float32 Adafactor with actual row/column variance factors.

    Matrix and higher-rank parameters factor the final two dimensions; vectors
    and scalars retain a full variance. Missing/frozen gradients skip all state
    and decay. Active empty gradients reject before mutation. eps[0]=None uses
    float32 machine epsilon. No foreach, graph capture or sparse gradients.
    """
    _adaptive = True
    _pair_options = _PAIR_OPTIONS + ('eps',)
    _all_state_names = ('row_var', 'col_var', 'variance')

    def __init__(self, params, lr=.01, beta2_decay=-.8, eps=(None, .001), d=1., weight_decay=0., *, maximize=False):
        self._initialize(params, dict(lr=lr, beta2_decay=beta2_decay, eps=tuple(eps), d=d, weight_decay=weight_decay, maximize=maximize))

    def _validate_group(self, group):
        if set(group) - (set(self.defaults) | {'params', *_GROUP_METADATA}) or not set(self.defaults) <= set(group):
            raise ValueError('unsupported or missing Adafactor group option')
        for key in ('lr', 'd', 'weight_decay'):
            _number(group[key], key)
        if group['d'] < 1:
            raise ValueError('Adafactor d must be at least one')
        decay=group['beta2_decay']
        if type(decay) not in (int,float) or not math.isfinite(decay) or decay > 0:
            raise ValueError('Adafactor beta2_decay must be finite and nonpositive')
        _number(abs(decay), 'abs(beta2_decay)')
        eps=group['eps']
        if not isinstance(eps,(tuple,list)) or len(eps)!=2:
            raise ValueError('Adafactor eps must be a pair')
        if eps[0] is not None:
            _number(eps[0], 'eps[0]')
        _number(eps[1], 'eps[1]')
        if type(group['maximize']) is not bool:
            raise ValueError('maximize must be bool')
        for key in _GROUP_METADATA:
            if key in group:
                _number(group[key], key)
        if any(_parameter_tensor(p).dtype!='float32' or _parameter_tensor(p).device!='cpu' for p in group['params']):
            raise ValueError('Adafactor requires CPU float32 parameters')
        limit = ((1 << (8 * struct.calcsize('P') - 1)) - 1) // 4
        for parameter in group['params']:
            for name in self._all_state_names:
                shape = self._state_shape(parameter, name)
                if math.prod(max(1, dim) for dim in shape) > limit:
                    raise ValueError('derived Adafactor factor exceeds addressable storage')

    def _config(self, group):
        self._validate_group(group)
        return [14., group['lr'], group['beta2_decay'], -1. if group['eps'][0] is None else group['eps'][0],
                group['eps'][1], group['d'], group['weight_decay'], float(group['maximize'])]

    def _new_native(self, group):
        return _Optimizer.adafactor(group['params'], self._config(group))

    def _state_names(self, group, parameter=None):
        return ('row_var','col_var') if len(_parameter_tensor(parameter).shape)>=2 else ('variance',)

    def _state_shape(self, parameter, name):
        shape=_parameter_tensor(parameter).shape
        if len(shape)>=2:
            if name=='variance':
                return [0]
            shape[-1 if name=='row_var' else -2]=1
            return shape
        return shape if name=='variance' else [0]

    def _commit_checkpoint_metadata(self, staged=None):
        if staged is None:
            groups=[]
            for current,c in zip(self.param_groups,self._native.configurations()):
                groups.append(dict(params=current['params'],lr=c[1],beta2_decay=c[2],eps=(None if c[3]==-1. else c[3],c[4]),
                                   d=c[5],weight_decay=c[6],maximize=bool(c[7])))
            staged=groups,[[True]*len(slots) for slots in self._slots]
        super()._commit_checkpoint_metadata(staged)


class LBFGS(_Base):
    """Native CPU float32 limited-memory BFGS with mandatory loss closure.

    One parameter group; fixed steps or line_search_fn='strong_wolfe'; no capture.
    All parameters must stay trainable; missing gradients are zero coordinates
    and coupled curvature history may still move them, matching Torch LBFGS.
    A closure must zero gradients, recompute a scalar loss and call backward.
    Successful steps retain the last closure's gradient handles, including a
    rejected line-search trial when it was evaluated last. Checkpoints require
    explicit zero_grad and cover complete calls only;
    a closure error after updates can leave partial progress. history_size is
    fixed for the constructed optimizer, including load_state_dict/restore.
    Repeated zero-rate searches keep finite parameters unchanged; Torch 2.13
    can instead raise ZeroDivisionError during zero-rate cubic interpolation.
    """
    _integers = ('max_iter', 'max_eval', 'history_size')

    def __init__(self, params, lr=1., max_iter=20, max_eval=None, tolerance_grad=1e-7,
                 tolerance_change=1e-9, history_size=100, line_search_fn=None):
        if type(max_iter) is not int:
            raise ValueError('max_iter must be an integer')
        self._busy = False
        self._history_size = history_size
        supplied = list(params)
        if supplied and isinstance(supplied[0], dict) and len(supplied) != 1:
            raise ValueError('LBFGS supports one parameter group')
        if supplied and isinstance(supplied[0], dict):
            self._history_size = supplied[0].get('history_size', history_size)
        self._initialize(supplied, dict(lr=lr, max_iter=max_iter, max_eval=max_iter * 5 // 4 if max_eval is None else max_eval,
                                       tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                                       history_size=history_size, line_search_fn=line_search_fn))

    def _validate_membership(self):
        if self._busy:
            raise RuntimeError('LBFGS state/configuration cannot be accessed during a closure')
        super()._validate_membership()

    def _validate_group(self, group):
        allowed = set(self.defaults) | {'params', *_GROUP_METADATA}
        if set(group) - allowed or not set(self.defaults) <= set(group):
            raise ValueError('unsupported or missing LBFGS group option')
        for key in ('lr', 'tolerance_grad', 'tolerance_change'):
            _number(group[key], key)
        for key in self._integers:
            if type(group[key]) is not int or not 1 <= group[key] <= 2**24:
                raise ValueError(f'{key} must be an integer in [1, 2**24]')
        if group['history_size'] != self._history_size:
            raise ValueError('LBFGS history_size must match the constructed optimizer')
        if group['line_search_fn'] is not None and (type(group['line_search_fn']) is not str or group['line_search_fn'] != 'strong_wolfe'):
            raise NotImplementedError('LBFGS supports line_search_fn=None or strong_wolfe')
        for key in _GROUP_METADATA:
            if key in group:
                _number(group[key], key)
        if any((isinstance(p, Parameter) and not p.trainable) or _parameter_tensor(p).device != 'cpu' or _parameter_tensor(p).dtype != 'float32' for p in group['params']):
            raise ValueError('LBFGS requires trainable CPU float32 parameters')
        if not sum(math.prod(_parameter_tensor(p).shape) for p in group['params']):
            raise ValueError('LBFGS requires at least one parameter element')

    def _config(self, group):
        self._validate_group(group)
        return [13., group['lr'], group['max_iter'], group['max_eval'], group['tolerance_grad'], group['tolerance_change'], group['history_size']] + ([1.] if group['line_search_fn'] is not None else [])

    def _new_native(self, group):
        return _Optimizer.lbfgs(group['params'], self._config(group))

    def add_param_group(self, param_group):
        raise ValueError('LBFGS supports one fixed parameter group')

    def step(self, closure):
        if not callable(closure):
            raise TypeError('LBFGS requires a callable loss closure')
        if _is_capturing():
            raise NotImplementedError('Python optimizer steps require eager execution')
        self._native.validate_gradient_storage()
        self._sync()
        from ._grad_mode import enable_grad
        def evaluate():
            with enable_grad():
                return closure()
        self._busy = True
        try:
            loss = _Optimizer.lbfgs_step(self._native, evaluate, preserve_grads=True)
        finally:
            self._busy = False
        self._seen = [[True] * len(self.params)]
        return loss

    def state_dict(self):
        self._sync()
        arrays = dict(self._native.state_tensors())
        header, scalars = arrays['header'].tolist(), arrays['scalars'].tolist()
        payload = {}
        if header[0]:
            payload = dict(func_evals=int(header[0]), n_iter=int(header[1]))
            if header[2]:
                count = int(header[4]); n = arrays['d'].shape[0]
                def rows(name):
                    values = arrays[name].reshape([arrays[name].numel()]).tolist()
                    return [Tensor(values[i*n:(i+1)*n], [n]) for i in range(count)]
                payload.update(d=arrays['d'], t=scalars[0], old_dirs=rows('old_dirs'), old_stps=rows('old_stps'),
                               ro=[Tensor([v], []) for v in arrays['rho'].tolist()[:count]], H_diag=Tensor([scalars[1]], []),
                               prev_flat_grad=arrays['prev_grad'], prev_loss=scalars[2])
                if header[3]:
                    payload['al'] = [Tensor([v], []) if present else None for v, present in zip(arrays['alpha'].tolist(), arrays['alpha_valid'].tolist())]
        group = {**self.param_groups[0], 'params': list(range(len(self.params)))}
        return {'state': {0: payload} if payload else {}, 'param_groups': [group]}

    def load_state_dict(self, state_dict):
        self._validate_membership()
        if type(state_dict) is not dict or set(state_dict) != {'state', 'param_groups'}:
            raise ValueError('LBFGS state needs state and param_groups')
        states, groups = state_dict['state'], state_dict['param_groups']
        if type(states) is not dict or not isinstance(groups, (list, tuple)) or len(groups) != 1 or type(groups[0]) is not dict:
            raise ValueError('LBFGS state/group schema mismatch')
        source = groups[0]; ids = source.get('params')
        if not isinstance(ids, (list, tuple)) or len(ids) != len(self.params) or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('LBFGS positional parameter IDs mismatch')
        if any(type(key) is not int or key != ids[0] for key in states):
            raise ValueError('LBFGS global history belongs to the first parameter only')
        group = {**source, 'params': list(self.params)}
        self._validate_group(group)
        payload = states.get(ids[0], {})
        n, h = sum(math.prod(_parameter_tensor(p).shape) for p in self.params), group['history_size']
        arrays = {'header': Tensor.zeros([5]), 'scalars': Tensor.zeros([3]), 'd': Tensor.zeros([n]), 'prev_grad': Tensor.zeros([n]),
                  'old_dirs': Tensor.zeros([h, n]), 'old_stps': Tensor.zeros([h, n]), 'rho': Tensor.zeros([h]),
                  'alpha': Tensor.zeros([h]), 'alpha_valid': Tensor.zeros([h])}
        if type(payload) is not dict:
            raise ValueError('LBFGS global state must be a dictionary')
        if payload:
            basic = {'func_evals', 'n_iter'}
            detail = {'d', 't', 'old_dirs', 'old_stps', 'ro', 'H_diag', 'prev_flat_grad', 'prev_loss'}
            expected = basic | (detail if 'd' in payload else set()) | ({'al'} if 'al' in payload else set())
            if set(payload) != expected or ('al' in payload and 'd' not in payload):
                raise ValueError('LBFGS state keys mismatch')
            for key in basic:
                if type(payload[key]) is not int or not 0 <= payload[key] <= 2**24:
                    raise ValueError('LBFGS counters must be exactly representable integers')
            def scalar(value):
                if isinstance(value, Tensor):
                    if value.dtype != 'float32' or value.device != 'cpu' or value.shape != []:
                        raise ValueError('LBFGS scalar state must be CPU float32')
                    return value.item()
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('LBFGS scalar state must be finite')
                return float(value)
            def vector(value):
                if not isinstance(value, Tensor) or value.dtype != 'float32' or value.device != 'cpu' or value.shape != [n]:
                    raise ValueError('LBFGS vector state shape/dtype/device mismatch')
                return value.tolist()
            count = 0
            if 'd' in payload:
                for key in ('old_dirs', 'old_stps', 'ro'):
                    if not isinstance(payload[key], (list, tuple)):
                        raise ValueError('LBFGS history must be a sequence')
                count = len(payload['old_dirs'])
                if count > h or len(payload['old_stps']) != count or len(payload['ro']) != count:
                    raise ValueError('LBFGS history length mismatch')
                arrays['d'], arrays['prev_grad'] = Tensor(vector(payload['d']), [n]), Tensor(vector(payload['prev_flat_grad']), [n])
                arrays['scalars'] = Tensor([scalar(payload[key]) for key in ('t', 'H_diag', 'prev_loss')], [3])
                for key in ('old_dirs', 'old_stps'):
                    values = [value for row in payload[key] for value in vector(row)]
                    arrays[key] = Tensor(values + [0.] * ((h-count)*n), [h, n])
                arrays['rho'] = Tensor([scalar(value) for value in payload['ro']] + [0.] * (h-count), [h])
                if 'al' in payload:
                    alpha = payload['al']
                    if not isinstance(alpha, (list, tuple)) or len(alpha) != h:
                        raise ValueError('LBFGS alpha cache length mismatch')
                    arrays['alpha'] = Tensor([0. if value is None else scalar(value) for value in alpha], [h])
                    arrays['alpha_valid'] = Tensor([float(value is not None) for value in alpha], [h])
            arrays['header'] = Tensor([payload['func_evals'], payload['n_iter'], float('d' in payload), float('al' in payload), count], [5])
        self._native.restore_state(self._config(group), list(arrays.items()))
        self.param_groups[0].clear(); self.param_groups[0].update(group)
        self._seen = [[bool(payload)] * len(self.params)]

    def _commit_checkpoint_metadata(self, staged=None):
        if staged is None:
            config = self._native.configurations()[0]
            group = dict(params=list(self.params), lr=config[1], max_iter=int(config[2]), max_eval=int(config[3]),
                         tolerance_grad=config[4], tolerance_change=config[5], history_size=int(config[6]), line_search_fn='strong_wolfe' if len(config) == 8 else None)
            staged = [group], [[bool(dict(self._native.state_tensors())['header'].tolist()[0])] * len(self.params)]
        super()._commit_checkpoint_metadata(staged)


class Muon(_Base):
    """Eager CPU float32 matrix optimizer with BF16-rounded Newton-Schulz updates.

    Parameters and momentum remain float32. The polynomial's matrix outputs,
    norm and normalization are explicitly rounded to BF16. No general BF16
    Tensor compute, graph capture, sparse gradients or GPU execution is implied.
    Epsilon must fit the finite BF16 range. Native scalar settings are float32;
    parity with Torch is numerical.
    """
    _pair_options=_PAIR_OPTIONS+('ns_coefficients',)
    _scalars=('lr','weight_decay','momentum','eps','ns_steps')
    _adjustments=(None,'original','match_rms_adamw')

    def __init__(self,params,lr=.001,weight_decay=.1,momentum=.95,nesterov=True,
                 ns_coefficients=(3.4445,-4.775,2.0315),eps=1e-7,ns_steps=5,adjust_lr_fn=None):
        self._initialize(params,dict(lr=lr,weight_decay=weight_decay,momentum=momentum,nesterov=nesterov,
            ns_coefficients=tuple(ns_coefficients),eps=eps,ns_steps=ns_steps,adjust_lr_fn=adjust_lr_fn))

    def _validate_group(self,group):
        if set(group)-(set(self.defaults)|{'params',*_GROUP_METADATA}) or not set(self.defaults)<=set(group):
            raise ValueError('unsupported or missing Muon group option')
        for key in self._scalars:
            _number(group[key],key)
        if group['eps']>float.fromhex('0x1.fep127'):
            raise ValueError('Muon eps exceeds finite BF16 normalization range')
        if type(group['ns_steps']) is not int or not 0<=group['ns_steps']<100:
            raise ValueError('ns_steps must be an integer from zero through 99')
        if type(group['nesterov']) is not bool:
            raise TypeError('nesterov must be bool')
        if (group['adjust_lr_fn'] is not None and type(group['adjust_lr_fn']) is not str) or group['adjust_lr_fn'] not in self._adjustments:
            raise ValueError('adjust_lr_fn must be None, original or match_rms_adamw')
        coefficients=group['ns_coefficients']
        if not isinstance(coefficients,(tuple,list)) or len(coefficients)!=3:
            raise ValueError('ns_coefficients must contain three finite scalars')
        for value in coefficients:
            if type(value) not in (int,float):raise TypeError('ns_coefficients must contain scalars')
            _number(abs(value),'abs(ns_coefficient)')
        for key in _GROUP_METADATA:
            if key in group:_number(group[key],key)
        limit=((1<<(8*struct.calcsize('P')-1))-1)//4
        for parameter in group['params']:
            tensor=_parameter_tensor(parameter);shape=tensor.shape
            if tensor.dtype!='float32' or tensor.device!='cpu' or len(shape)!=2 or shape[1]==0:
                raise ValueError('Muon requires CPU float32 matrix parameters with nonzero columns')
            if math.prod(shape)>limit or min(shape)**2>limit:
                raise ValueError('Muon work array exceeds addressable storage')

    def _config(self,group):
        self._validate_group(group)
        return [15.,group['lr'],group['weight_decay'],group['momentum'],float(group['nesterov']),
                *group['ns_coefficients'],group['eps'],float(group['ns_steps']),float(self._adjustments.index(group['adjust_lr_fn']))]

    def _new_native(self,group):
        return _Optimizer.muon(group['params'],self._config(group))

    def state_dict(self):
        self._sync();arrays=dict(self._native.state_tensors());state,groups,index={},[],0
        for gi,(group,slots) in enumerate(zip(self.param_groups,self._slots)):
            prefix,ids=self._prefix(gi),[]
            for pi,parameter in enumerate(slots):
                ids.append(index)
                if arrays[prefix+f'initialized.{pi}'].item():
                    state[index]={'momentum_buffer':arrays[prefix+f'momentum_buffer.{pi}'].reshape(_parameter_tensor(parameter).shape)}
                index+=1
            groups.append({**{key:tuple(value) if key=='ns_coefficients' else value for key,value in group.items() if key!='params'},'params':ids})
        return dict(state=state,param_groups=groups)

    def load_state_dict(self,state_dict):
        self._validate_membership()
        if type(state_dict) is not dict or set(state_dict)!={'state','param_groups'}:
            raise ValueError('optimizer state needs state and param_groups')
        states,saved=state_dict['state'],state_dict['param_groups']
        if type(states) is not dict or not isinstance(saved,(tuple,list)) or len(saved)!=len(self._slots):
            raise ValueError('optimizer state/group count mismatch')
        groups,arrays,seen,all_ids=[],[],[],set()
        for gi,(source,slots) in enumerate(zip(saved,self._slots)):
            if type(source) is not dict or not isinstance(source.get('params'),(tuple,list)) or len(source['params'])!=len(slots):
                raise ValueError('optimizer group parameter count mismatch')
            ids=source['params']
            if any(type(i) is not int or i<0 or i in all_ids for i in ids) or len(set(ids))!=len(ids):
                raise ValueError('optimizer state IDs must be unique nonnegative integers')
            all_ids.update(ids)
            group={**source,'params':list(slots)}
            self._validate_group(group);group['ns_coefficients']=tuple(group['ns_coefficients']);groups.append(group)
            flags=[];prefix=self._prefix(gi)
            for pi,(identifier,parameter) in enumerate(zip(ids,slots)):
                payload=states.get(identifier,{})
                if type(payload) is not dict or (payload and set(payload)!={'momentum_buffer'}):
                    raise ValueError('Muon slot state requires only momentum_buffer')
                initialized=bool(payload);flags.append(initialized)
                value=payload.get('momentum_buffer')
                if value is None and initialized:
                    raise ValueError('Muon momentum buffer must be a Tensor')
                if initialized and (not isinstance(value,Tensor) or value.dtype!='float32' or value.device!='cpu' or value.shape!=_parameter_tensor(parameter).shape):
                    raise ValueError('Muon momentum shape/dtype/device mismatch')
                n=math.prod(_parameter_tensor(parameter).shape)
                arrays.extend([(prefix+f'initialized.{pi}',Tensor([float(initialized)],[])),
                               (prefix+f'momentum_buffer.{pi}',value.reshape([n]) if initialized else Tensor.zeros([n]))])
            seen.append(flags)
        if any(type(key) is not int or key not in all_ids for key in states):
            raise ValueError('optimizer state contains an unknown parameter ID')
        self._native.restore_state(self._packed_config(groups),arrays)
        for target,source in zip(self.param_groups,groups):target.clear();target.update(source)
        self._seen=seen

    def _checkpoint_metadata(self):
        self._validate_membership();values={}
        encode=lambda value:struct.unpack('<Q',struct.pack('<d',value))[0]
        for gi,(group,seen) in enumerate(zip(self.param_groups,self._seen)):
            self._validate_group(group)
            for key in self._scalars:values[f'g{gi}.{key}']=encode(group[key])
            for i,value in enumerate(group['ns_coefficients']):values[f'g{gi}.ns_coefficients.{i}']=encode(value)
            values[f'g{gi}.nesterov']=int(group['nesterov'])
            values[f'g{gi}.adjust_lr_fn']=self._adjustments.index(group['adjust_lr_fn'])
            for key in _GROUP_METADATA:
                values[f'g{gi}.{key}.present']=int(key in group);values[f'g{gi}.{key}']=encode(group.get(key,0.))
            for pi,flag in enumerate(seen):values[f'g{gi}.seen.{pi}']=int(flag)
        return values

    def _prepare_checkpoint_metadata(self,metadata,native_configuration):
        if set(metadata)!=set(self._checkpoint_metadata()):raise ValueError('optimizer metadata/group schema mismatch')
        decode=lambda value:struct.unpack('<d',struct.pack('<Q',value))[0]
        groups,seen=[],[]
        for gi,slots in enumerate(self._slots):
            group={'params':list(slots)}
            for key in self._scalars:group[key]=decode(metadata[f'g{gi}.{key}'])
            steps=group['ns_steps']
            if not math.isfinite(steps) or steps!=int(steps):raise ValueError('invalid saved Muon iteration count')
            group['ns_steps']=int(steps)
            group['ns_coefficients']=tuple(decode(metadata[f'g{gi}.ns_coefficients.{i}']) for i in range(3))
            flag=metadata[f'g{gi}.nesterov'];adjustment=metadata[f'g{gi}.adjust_lr_fn']
            if flag not in (0,1) or adjustment not in (0,1,2):raise ValueError('invalid saved Muon configuration flag')
            group['nesterov']=bool(flag);group['adjust_lr_fn']=self._adjustments[adjustment]
            for key in _GROUP_METADATA:
                present=metadata[f'g{gi}.{key}.present'];value=decode(metadata[f'g{gi}.{key}'])
                if present not in (0,1):raise ValueError('invalid scheduler metadata presence flag')
                if present:group[key]=value
                elif value!=0.:raise ValueError('unexpected inactive scheduler metadata value')
            self._validate_group(group);groups.append(group)
            flags=[metadata[f'g{gi}.seen.{pi}'] for pi in range(len(slots))]
            if any(flag not in (0,1) for flag in flags):raise ValueError('invalid optimizer initialized flag')
            seen.append([bool(flag) for flag in flags])
        expected=self._packed_config(groups)
        if native_configuration.dtype!='float32' or native_configuration.shape!=[len(expected)]:
            raise ValueError('optimizer configuration shape/dtype mismatch')
        if struct.pack('<'+'f'*len(expected),*expected)!=struct.pack('<'+'f'*len(expected),*native_configuration.tolist()):
            raise ValueError('optimizer metadata disagrees with native configuration')
        return groups,seen

    def _commit_checkpoint_metadata(self,staged=None):
        if staged is None:
            groups=[];seen=[];arrays=dict(self._native.state_tensors())
            for gi,(current,c) in enumerate(zip(self.param_groups,self._native.configurations())):
                groups.append(dict(params=current['params'],lr=c[1],weight_decay=c[2],momentum=c[3],nesterov=bool(c[4]),
                                   ns_coefficients=tuple(c[5:8]),eps=c[8],ns_steps=int(c[9]),adjust_lr_fn=self._adjustments[int(c[10])]))
                seen.append([bool(arrays[self._prefix(gi)+f'initialized.{pi}'].item()) for pi in range(len(current['params']))])
            staged=groups,seen
        super()._commit_checkpoint_metadata(staged)


from . import lr_scheduler  # noqa: E402
import sys as _sys
_sys.modules[__name__ + '.lr_scheduler'] = lr_scheduler
