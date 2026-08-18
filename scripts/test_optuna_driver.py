"""Проверки драйвера Optuna без Isaac Sim: YAML, argv, OPTUNA_METRIC_FILE."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rl.optuna_driver import (  # noqa: E402
    BAD_METRIC_SENTINEL,
    build_train_argv,
    load_optuna_config,
    run_study,
    run_trial,
    suggest_params,
)


class FakeTrial:
    """Минимальный trial: suggest_* возвращает нижнюю границу / первый choice."""

    def suggest_float(self, name, low, high, log=False, step=None):
        return float(low)

    def suggest_int(self, name, low, high, log=False, step=1):
        return int(low)

    def suggest_categorical(self, name, choices):
        return list(choices)[0]


def _write_fake_train(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


class TestLoadSearchSpace(unittest.TestCase):
    def test_ann_yaml_matches_legacy_bounds(self):
        cfg = load_optuna_config(_REPO_ROOT / "configs" / "optuna" / "ann.yaml")
        self.assertEqual(cfg["agent"], "ann")
        self.assertEqual(cfg["max_steps"], 50000)
        space = cfg["search_space"]
        self.assertEqual(space["ppo.lr"]["low"], 1e-5)
        self.assertEqual(space["ppo.lr"]["high"], 1e-3)
        self.assertTrue(space["ppo.lr"]["log"])
        self.assertEqual(space["ppo.num_steps"]["low"], 16)
        self.assertEqual(space["ppo.num_steps"]["high"], 64)
        self.assertEqual(space["ppo.mini_batch_size"]["low"], 64)
        self.assertEqual(space["ppo.mini_batch_size"]["high"], 512)
        self.assertEqual(space["ppo.ppo_epochs"]["low"], 3)
        self.assertEqual(space["ppo.ppo_epochs"]["high"], 15)
        self.assertEqual(space["ppo.clip_param"]["low"], 0.1)
        self.assertEqual(space["ppo.clip_param"]["high"], 0.3)
        self.assertNotIn("agent.T", space)

    def test_snn_and_hybrid_yaml_match_legacy_bounds(self):
        for name in ("snn", "hybrid"):
            cfg = load_optuna_config(_REPO_ROOT / "configs" / "optuna" / ("%s.yaml" % name))
            self.assertEqual(cfg["agent"], name)
            self.assertEqual(cfg["max_steps"], 10000)
            space = cfg["search_space"]
            self.assertEqual(space["ppo.mini_batch_size"]["low"], 128)
            self.assertEqual(space["agent.T"]["low"], 1)
            self.assertEqual(space["agent.T"]["high"], 10)
            self.assertEqual(space["agent.alpha"]["low"], 0.2)
            self.assertEqual(space["agent.alpha"]["high"], 0.8)
            self.assertEqual(space["agent.lif_v_th"]["low"], 0.2)
            self.assertEqual(space["agent.lif_v_th"]["high"], 0.6)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_optuna_config(_REPO_ROOT / "configs" / "optuna" / "missing.yaml")

    def test_reserved_keys_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("search_space:\n  seed:\n    type: int\n    low: 0\n    high: 1\n")
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                load_optuna_config(path)
        finally:
            os.remove(path)


class TestSuggestAndArgv(unittest.TestCase):
    def test_suggest_params_uses_hydra_keys(self):
        cfg = load_optuna_config(_REPO_ROOT / "configs" / "optuna" / "snn.yaml")
        params = suggest_params(FakeTrial(), cfg["search_space"])
        self.assertEqual(params["ppo.lr"], 1e-5)
        self.assertEqual(params["agent.T"], 1)
        self.assertEqual(params["agent.alpha"], 0.2)

    def test_argv_is_hydra_overrides_plus_headless(self):
        argv = build_train_argv(
            "/usr/bin/python",
            "/tmp/train.py",
            "snn",
            {"ppo.lr": 0.00031, "agent.T": 4, "agent.alpha": 0.5},
            task="Isaac-Velocity-Flat-Unitree-A1-v0",
            seed=2,
            run_name="trial_1",
            max_steps=10000,
            num_envs=64,
            use_mlflow=False,
        )
        self.assertEqual(argv[0], "/usr/bin/python")
        self.assertEqual(argv[1], "/tmp/train.py")
        self.assertIn("agent=snn", argv)
        self.assertIn("--headless", argv)
        self.assertIn("task=Isaac-Velocity-Flat-Unitree-A1-v0", argv)
        self.assertIn("seed=2", argv)
        self.assertIn("run_name=trial_1", argv)
        self.assertIn("ppo.max_steps=10000", argv)
        self.assertIn("num_envs=64", argv)
        self.assertIn("use_mlflow=false", argv)
        self.assertIn("ppo.lr=0.00031", argv)
        self.assertIn("agent.T=4", argv)
        self.assertIn("agent.alpha=0.5", argv)

    def test_use_mlflow_true_override(self):
        argv = build_train_argv(
            "python",
            "train.py",
            "ann",
            {},
            task="t",
            seed=1,
            run_name="trial_0",
            max_steps=None,
            num_envs=None,
            use_mlflow=True,
            headless=False,
        )
        self.assertNotIn("--headless", argv)
        self.assertIn("use_mlflow=true", argv)
        self.assertTrue(all(not a.startswith("ppo.max_steps=") for a in argv))
        self.assertTrue(all(not a.startswith("num_envs=") for a in argv))

    def test_extra_overrides_before_trial_params(self):
        argv = build_train_argv(
            "python",
            "train.py",
            "snn",
            {"ppo.lr": 1e-4},
            task="t",
            seed=1,
            run_name="trial_0",
            max_steps=None,
            num_envs=None,
            use_mlflow=False,
            extra_overrides=["ppo.gamma=0.9"],
        )
        self.assertIn("ppo.gamma=0.9", argv)
        self.assertLess(argv.index("ppo.gamma=0.9"), argv.index("ppo.lr=0.0001"))


class TestRunTrialMetricFile(unittest.TestCase):
    def test_reads_metric_written_by_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_script = Path(tmp) / "fake_train.py"
            argv_dump = Path(tmp) / "argv.txt"
            _write_fake_train(
                train_script,
                """
                import os, sys
                from pathlib import Path
                path = os.environ["OPTUNA_METRIC_FILE"]
                Path(path).write_text("12.500000\\n", encoding="utf-8")
                Path(%r).write_text("\\n".join(sys.argv[1:]), encoding="utf-8")
                """
                % str(argv_dump),
            )
            metric = run_trial(
                {"ppo.lr": 3e-4, "agent.T": 5},
                agent="snn",
                task="Fake-Task",
                seed=3,
                run_name="trial_0",
                max_steps=100,
                num_envs=8,
                use_mlflow=False,
                script_dir=tmp,
                python_exe=sys.executable,
                train_script=str(train_script),
            )
            self.assertAlmostEqual(metric, 12.5)
            dumped = argv_dump.read_text(encoding="utf-8").splitlines()
            self.assertIn("agent=snn", dumped)
            self.assertIn("--headless", dumped)
            self.assertIn("ppo.lr=0.0003", dumped)
            self.assertIn("agent.T=5", dumped)

    def test_crash_without_metric_returns_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_script = Path(tmp) / "fake_train.py"
            _write_fake_train(
                train_script,
                """
                import sys
                sys.exit(1)
                """,
            )
            metric = run_trial(
                {},
                agent="ann",
                task="Fake-Task",
                seed=1,
                run_name="trial_0",
                max_steps=None,
                num_envs=None,
                use_mlflow=False,
                script_dir=tmp,
                python_exe=sys.executable,
                train_script=str(train_script),
            )
            self.assertEqual(metric, BAD_METRIC_SENTINEL)

    def test_crash_with_sentinel_file_returns_that_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_script = Path(tmp) / "fake_train.py"
            _write_fake_train(
                train_script,
                """
                import os, sys
                path = os.environ["OPTUNA_METRIC_FILE"]
                with open(path, "w", encoding="utf-8") as f:
                    f.write("-1000000000.000000\\n")
                sys.exit(1)
                """,
            )
            metric = run_trial(
                {},
                agent="ann",
                task="Fake-Task",
                seed=1,
                run_name="trial_0",
                max_steps=None,
                num_envs=None,
                use_mlflow=False,
                script_dir=tmp,
                python_exe=sys.executable,
                train_script=str(train_script),
            )
            self.assertEqual(metric, BAD_METRIC_SENTINEL)

    def test_sets_optuna_metric_file_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_script = Path(tmp) / "fake_train.py"
            env_dump = Path(tmp) / "env.txt"
            _write_fake_train(
                train_script,
                """
                import os
                from pathlib import Path
                dump = Path(%r)
                dump.write_text(os.environ.get("OPTUNA_METRIC_FILE", ""), encoding="utf-8")
                Path(os.environ["OPTUNA_METRIC_FILE"]).write_text("0.0\\n", encoding="utf-8")
                """
                % str(env_dump),
            )
            run_trial(
                {},
                agent="ann",
                task="Fake-Task",
                seed=1,
                run_name="trial_0",
                max_steps=None,
                num_envs=None,
                use_mlflow=False,
                script_dir=tmp,
                python_exe=sys.executable,
                train_script=str(train_script),
            )
            metric_path = env_dump.read_text(encoding="utf-8").strip()
            self.assertTrue(metric_path.endswith(".optuna_metric"))
            # файл метрики удаляется в finally драйвера
            self.assertFalse(os.path.isfile(metric_path))


def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_optuna_available(), "optuna не установлен")
class TestRunStudy(unittest.TestCase):
    def test_study_maximizes_metric_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_script = Path(tmp) / "fake_train.py"
            _write_fake_train(
                train_script,
                """
                import os
                path = os.environ["OPTUNA_METRIC_FILE"]
                with open(path, "w", encoding="utf-8") as f:
                    f.write("3.0\\n")
                """,
            )
            search_space = {
                "ppo.lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}
            }
            study = run_study(
                agent="ann",
                search_space=search_space,
                n_trials=2,
                task="Fake-Task",
                seed=1,
                max_steps=10,
                num_envs=None,
                use_mlflow=False,
                study_name="unit-test",
                script_dir=tmp,
                python_exe=sys.executable,
                train_script=str(train_script),
            )
            self.assertAlmostEqual(study.best_value, 3.0)
            self.assertEqual(len(study.trials), 2)
            self.assertIn("ppo.lr", study.best_params)


class TestLazyPackageImport(unittest.TestCase):
    def test_optuna_driver_import_does_not_load_torch(self):
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "import rl.optuna_driver\n"
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:20]\n"
            "assert 'norse' not in sys.modules\n"
            "assert 'isaaclab' not in sys.modules\n"
        ) % str(_SCRIPTS_DIR)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
