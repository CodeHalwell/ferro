"""Opt-in RED public API contracts: run this file with the ferro project Python.

Not named test_*.py: unfinished facade contracts must not join default discovery.
No mocks, skips, private imports, or implementation are provided here.
"""
import importlib
import unittest


class PythonAPIContract(unittest.TestCase):
    def api(self, name):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            self.fail(f"RED: required public module {name!r} is unavailable: {exc}")

    def scalar_model(self):
        ferro, nn = self.api("ferro"), self.api("ferro.nn")

        class Scale(nn.Module):
            initial: float = 1.0

            def build(self):
                self.weight = nn.Parameter(ferro.Tensor([self.initial], [1]))

            def forward(self, x):
                return x * self.weight.tensor()

        return Scale

    def test_annotated_config_build_once_and_fresh_parameters(self):
        ferro, nn = self.api("ferro"), self.api("ferro.nn")
        builds = []

        class Dense(nn.Module):
            in_features: int
            out_features: int = 2

            def build(self):
                builds.append(self)
                self.proj = nn.Linear(self.in_features, self.out_features)

            def forward(self, x):
                return self.proj(x)

        a, b = Dense(in_features=3), Dense(in_features=5, out_features=4)
        self.assertEqual(list(a(ferro.Tensor.ones([2, 3])).shape), [2, 2])
        self.assertEqual(list(b(ferro.Tensor.ones([2, 5])).shape), [2, 4])
        a.eval().train().to("cpu")
        a(ferro.Tensor.ones([1, 3]))
        self.assertEqual(builds, [a, b])
        self.assertEqual(list(a.proj.weight.tensor().shape), [2, 3])
        self.assertEqual(list(b.proj.weight.tensor().shape), [4, 5])
        c = Dense(in_features=3)
        self.assertIsNot(a.proj.weight, c.proj.weight)
        old = c.proj.weight.tensor().tolist()
        with ferro.no_grad():
            a.proj.weight.tensor().copy_(ferro.Tensor.zeros([2, 3]))
        self.assertEqual(c.proj.weight.tensor().tolist(), old)

    def test_constructor_rejects_missing_unknown_and_unannotated_configuration(self):
        nn = self.api("ferro.nn")
        builds = []

        class Configured(nn.Module):
            width: int
            ordinary_constant = 7

            def build(self):
                builds.append(self.width)

        with self.assertRaises(TypeError):
            Configured()
        with self.assertRaises(TypeError):
            Configured(width=2, ordinary_constant=9)
        with self.assertRaises(TypeError):
            Configured(width=2, typo=9)
        self.assertEqual(builds, [])
        self.assertEqual(Configured(width=3).width, 3)
        self.assertEqual(builds, [3])

    def test_inherited_config_build_dispatches_once(self):
        nn = self.api("ferro.nn")
        calls = []

        class Base(nn.Module):
            width: int

            def build(self):
                calls.append("base")

        class Derived(Base):
            depth: int = 2

            def build(self):
                calls.append((self.width, self.depth))

        Derived(width=3)
        self.assertEqual(calls, [(3, 2)])

    def test_live_class_parameters_are_rejected(self):
        ferro, nn = self.api("ferro"), self.api("ferro.nn")
        with self.assertRaisesRegex(TypeError, "class|shared|build"):
            class Unsafe(nn.Module):
                weight = nn.Parameter(ferro.Tensor.ones([1]))

    def test_nested_registration_buffers_and_tied_update_once(self):
        ferro, nn, optim = self.api("ferro"), self.api("ferro.nn"), self.api("ferro.optim")
        Scale = self.scalar_model()

        class Tied(nn.Module):
            def build(self):
                self.left = Scale()
                self.right = self.left
                self.register_buffer("mask", ferro.Tensor.ones([1]))

            def forward(self, x):
                return self.left(x) + self.right(x)

        m = Tied()
        named = dict(m.named_parameters())
        self.assertEqual(list(named), ["left.weight"])
        self.assertIs(named["left.weight"], m.left.weight)
        self.assertEqual(list(dict(m.named_buffers())), ["mask"])
        self.assertIs(dict(m.named_buffers())["mask"], m.mask)
        optimizer = optim.SGD(m.parameters(), lr=0.1)
        m(ferro.Tensor.ones([1])).sum().backward()
        optimizer.step()
        self.assertAlmostEqual(m.left.weight.tensor().item(), 0.8, places=6)
        self.assertEqual(m.mask.tolist(), [1.0])

    def test_registered_containers_and_replacement(self):
        nn = self.api("ferro.nn")
        Scale = self.scalar_model()

        class Stack(nn.Module):
            def build(self):
                self.layers = nn.ModuleList([Scale(), Scale()])

        m = Stack()
        self.assertEqual(list(dict(m.named_parameters())), ["layers.0.weight", "layers.1.weight"])
        replacement = Scale(initial=3.0)
        m.layers[0] = replacement
        self.assertIs(dict(m.named_parameters())["layers.0.weight"], replacement.weight)
        m.eval()
        self.assertFalse(m.layers[0].training)
        self.assertFalse(m.layers[1].training)

    def test_optimizer_keeps_parameter_identity_across_cpu_placement(self):
        ferro, optim = self.api("ferro"), self.api("ferro.optim")
        m = self.scalar_model()()
        parameter = m.weight
        optimizer = optim.SGD(m.parameters(), lr=0.1)
        self.assertIs(m.to("cpu"), m)
        self.assertIs(m.weight, parameter)
        self.assertIs(list(m.parameters())[0], parameter)
        m(ferro.Tensor.ones([1])).sum().backward()
        optimizer.step()
        self.assertAlmostEqual(parameter.tensor().item(), 0.9, places=6)
        self.assertEqual(parameter.tensor().device, "cpu")

    def test_eval_mode_keeps_input_gradients(self):
        ferro = self.api("ferro")
        m = self.scalar_model()(initial=2.0).eval()
        x = ferro.Tensor([3.0], [1]).requires_grad_(True)
        m(x).sum().backward()
        self.assertFalse(m.training)
        self.assertAlmostEqual(x.grad.item(), 2.0, places=6)

    def test_nested_no_grad_restores_on_exception(self):
        ferro = self.api("ferro")
        self.assertTrue(callable(getattr(ferro, "no_grad", None)), "RED: missing public ferro.no_grad context")
        self.assertTrue(callable(getattr(ferro, "enable_grad", None)), "RED: missing public ferro.enable_grad context")
        x = ferro.Tensor.ones([1]).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with ferro.no_grad():
                self.assertFalse((x * x).requires_grad)
                with ferro.enable_grad():
                    self.assertTrue((x * x).requires_grad)
                self.assertFalse((x * x).requires_grad)
                raise RuntimeError("sentinel")
        (x * x).sum().backward()
        self.assertAlmostEqual(x.grad.item(), 2.0, places=6)

    def test_supervised_trainer_matches_explicit_step(self):
        ferro, optim = self.api("ferro"), self.api("ferro.optim")
        losses, training = self.api("ferro.nn.losses"), self.api("ferro.training")
        Scale = self.scalar_model()
        automatic, explicit = Scale(), Scale()
        x, y = ferro.Tensor([1.0, 2.0], [2, 1]), ferro.Tensor([2.0, 4.0], [2, 1])
        loss_fn = losses.MSELoss(reduction="mean")
        trainer = training.Trainer(automatic, optimizer=optim.SGD(automatic.parameters(), lr=0.1), loss_fn=loss_fn)
        manual_optimizer = optim.SGD(explicit.parameters(), lr=0.1)
        for _ in range(3):
            manual_optimizer.zero_grad()
            loss_fn(explicit(x), y).backward()
            manual_optimizer.step()
        trainer.fit([(x, y)], epochs=3)
        self.assertAlmostEqual(automatic.weight.tensor().item(), explicit.weight.tensor().item(), places=6)
        self.assertLess(loss_fn(automatic(x), y).item(), loss_fn(x, y).item())

    def test_custom_step_owns_multioptimizer_updates_and_batch_identity(self):
        ferro, optim, training = self.api("ferro"), self.api("ferro.optim"), self.api("ferro.training")
        left, right = self.scalar_model()(), self.scalar_model()()
        first, second = optim.SGD(left.parameters(), lr=0.1), optim.SGD(right.parameters(), lr=0.2)
        batch = {"coordinates": ferro.Tensor.ones([1]), "state": object()}
        seen = []

        def step(model, supplied):
            self.assertIs(model, left)
            self.assertIs(supplied, batch)
            seen.append(supplied)
            first.zero_grad()
            second.zero_grad()
            loss = (left(supplied["coordinates"]) + right(supplied["coordinates"])).sum()
            loss.backward()
            first.step()
            second.step()
            return {"loss": loss}

        training.Trainer(left, train_step=step).fit([batch], epochs=1)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(left.weight.tensor().item(), 0.9, places=6)
        self.assertAlmostEqual(right.weight.tensor().item(), 0.8, places=6)

    def test_objective_trainer_owns_exactly_one_update_and_keeps_metrics_live(self):
        ferro, optim, training = self.api("ferro"), self.api("ferro.optim"), self.api("ferro.training")
        m = self.scalar_model()()
        x = ferro.Tensor.ones([1])
        observed = []

        def objective(model, batch):
            self.assertIs(batch, x)
            loss = model(batch).sum()
            observed.append(loss)
            return loss

        def metrics(model, batch, loss):
            self.assertIs(loss, observed[-1])
            return {"raw_loss": loss}

        trainer = training.Trainer(m, optimizer=optim.SGD(m.parameters(), lr=0.1), objective=objective, metrics=metrics)
        reports = trainer.fit([x], epochs=2)
        self.assertAlmostEqual(m.weight.tensor().item(), 0.8, places=6)
        self.assertEqual(len(observed), 2)
        self.assertIs(reports[0]["loss"], observed[0])
        self.assertIs(reports[0]["raw_loss"], observed[0])

    def test_trainer_rejects_ambiguous_update_ownership_before_execution(self):
        optim, losses, training = self.api("ferro.optim"), self.api("ferro.nn.losses"), self.api("ferro.training")
        m = self.scalar_model()()
        optimizer = optim.SGD(m.parameters(), lr=0.1)
        objective = lambda model, batch: model(batch).sum()
        custom = lambda model, batch: {"loss": model(batch).sum()}
        for kwargs in (
            {"loss_fn": losses.MSELoss(), "objective": objective, "optimizer": optimizer},
            {"objective": objective, "train_step": custom},
            {"loss_fn": losses.MSELoss(), "train_step": custom},
            {"train_step": custom, "optimizer": optimizer},
            {"objective": objective},
        ):
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaises((TypeError, ValueError)):
                    training.Trainer(m, **kwargs)
        self.assertEqual(m.weight.tensor().item(), 1.0)

    def test_trainer_restores_each_nested_mode_after_fit_error(self):
        nn, training = self.api("ferro.nn"), self.api("ferro.training")
        Scale = self.scalar_model()

        class Pair(nn.Module):
            def build(self):
                self.a, self.b = Scale(), Scale()

        m = Pair().train()
        m.b.eval()

        def fail(model, batch):
            self.assertTrue(model.training)
            self.assertTrue(model.b.training)
            raise RuntimeError("sentinel")

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            training.Trainer(m, train_step=fail).fit([object()], epochs=1)
        self.assertTrue(m.training)
        self.assertTrue(m.a.training)
        self.assertFalse(m.b.training)

    def test_gradient_enabled_evaluation_returns_live_results_and_restores_mode(self):
        ferro, training = self.api("ferro"), self.api("ferro.training")
        m = self.scalar_model()(initial=2.0).train()
        x = ferro.Tensor([3.0], [1]).requires_grad_(True)
        outputs = []

        def evaluate(model, batch):
            self.assertFalse(model.training)
            result = model(batch).sum()
            outputs.append(result)
            return {"residual": result}

        result = training.Trainer(m, eval_step=evaluate).evaluate([x], grad_enabled=True)
        self.assertTrue(m.training)
        self.assertIs(result[0]["residual"], outputs[0])
        result[0]["residual"].backward()
        self.assertAlmostEqual(x.grad.item(), 2.0, places=6)

    def test_evaluation_error_restores_mode_and_grad_context(self):
        ferro, training = self.api("ferro"), self.api("ferro.training")
        m = self.scalar_model()().train()
        x = ferro.Tensor.ones([1]).requires_grad_(True)

        def fail(model, batch):
            self.assertFalse(model.training)
            self.assertFalse(model(batch).requires_grad)
            raise RuntimeError("sentinel")

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            training.Trainer(m, eval_step=fail).evaluate([x], grad_enabled=False)
        self.assertTrue(m.training)
        m(x).sum().backward()
        self.assertAlmostEqual(x.grad.item(), 1.0, places=6)

    def test_recurrent_state_is_not_reset_by_modes_or_trainer(self):
        ferro, nn, training = self.api("ferro"), self.api("ferro.nn"), self.api("ferro.training")

        class Accumulator(nn.Module):
            def build(self):
                self.state = ferro.Tensor.zeros([1])

            def forward(self, x):
                self.state = self.state + x
                return self.state

            def reset_state(self):
                self.state = ferro.Tensor.zeros([1])

        m = Accumulator()
        m(ferro.Tensor.ones([1]))
        m.eval().train()
        training.Trainer(m, train_step=lambda model, batch: {"value": model(batch)}).fit([ferro.Tensor.ones([1])], epochs=2)
        self.assertEqual(m.state.tolist(), [3.0])
        m.reset_state()
        self.assertEqual(m.state.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
