from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
from crabnet.utils.composition import _fractional_composition_L, _element_composition_L
from mat_discover.mat_discover_ import Discover, my_mvn


class Adapt(Discover):
    def __init__(self, train_df, val_df, **Discover_kwargs):
        super().__init__(**Discover_kwargs)
        self.train_df = deepcopy(train_df)
        self.val_df = deepcopy(val_df)

    def suggest_first_experiment(
        self,
        proxy_name="density",
        random_search=False,
        print_experiment=True,
        **predict_kwargs,
    ):
        first_experiment = self.suggest_next_experiment(
            proxy_name=proxy_name,
            fit=True,
            predict=True,
            random_search=random_search,
            print_experiment=print_experiment,
            **predict_kwargs,
        )
        self.init_pred_scaler = deepcopy(self.pred_scaler)
        self.init_proxy_scaler = deepcopy(self.proxy_scaler)
        return first_experiment

    def suggest_next_experiment(
        self,
        proxy_name="density",
        fit=True,
        predict=False,
        random_search=False,
        print_experiment=True,
        fit_verbose=False,
        **predict_kwargs,
    ):
        if not random_search:
            if fit:
                self.fit(self.train_df, verbose=fit_verbose, save=False)
            elif self.crabnet_model is None:
                raise ValueError(
                    "Run `disc.fit(train_df)` method or specify `fit_afresh=True`."
                )
            if predict:
                # TODO: precompute dm, umap, etc.
                self.predict(self.val_df, **predict_kwargs)
            else:
                if self.crabnet_model is not None:
                    val_true, self.val_pred, _, val_sigma = self.crabnet_model.predict(
                        self.val_df
                    )
                else:
                    self.val_pred = np.zeros(self.val_df.shape[0])
                # convert back to NumPy arrays
                if self.proxy_weight != 0:
                    train_emb = np.array(self.train_df.emb.tolist())
                    val_emb = np.array(self.val_df.emb.tolist())

                    train_r_orig = self.train_df.r_orig.values
                    mvn_list = list(
                        map(my_mvn, train_emb[:, 0], train_emb[:, 1], train_r_orig)
                    )
                    pdf_list = [mvn.pdf(val_emb) for mvn in mvn_list]

                    self.val_dens = np.sum(pdf_list, axis=0)
                    self.val_log_dens = np.log(self.val_dens)

                    self.val_df["emb"] = list(map(tuple, val_emb))
                    self.val_df.loc[:, "dens"] = self.val_dens
                else:
                    self.val_dens = np.zeros(self.val_df.shape[0])
                    self.val_df["emb"] = list(
                        map(tuple, np.zeros((self.val_df.shape[0], 2)).tolist())
                    )
                    self.val_df["dens"] = self.val_dens
                # recompute dens score
                # TODO: use init_pred_scaler and init_proxy_scaler
                self.dens_score = self.weighted_score(
                    self.val_pred,
                    self.val_dens,
                    pred_weight=self.pred_weight,
                    proxy_weight=self.proxy_weight,
                    pred_scaler=self.init_pred_scaler,
                    proxy_scaler=self.init_proxy_scaler,
                )
            self.dens_score_df = self.sort(self.dens_score)

            proxy_lookup = {
                "density": "dens_score_df",
                "peak": "peak_score_df",
                "radius": "rad_score_df",
            }
            proxy_df_name = proxy_lookup[proxy_name]
            proxy_df = getattr(self, proxy_df_name)
            next_formula, next_index, next_proxy, next_score = [
                proxy_df[name].values[0]
                for name in ["formula", "index", proxy_name, "score"]
            ]
            next_target, next_emb, next_dens = [
                self.val_df[self.val_df["index"] == next_index][name].values[0]
                for name in ["target", "emb", "dens"]
            ]
        else:
            sample = self.val_df.sample(1)
            next_formula, next_target, next_index = sample[
                ["formula", "target", "index"]
            ].values[0]
            next_proxy = np.nan
            next_score = np.nan
            next_emb = np.nan
            next_dens = np.nan

        next_experiment = {
            "formula": next_formula,
            "index": next_index,
            "target": next_target,
            "emb": next_emb,
            "dens": next_dens,
        }

        # append compound to train, remove from val, and reset indices
        # https://stackoverflow.com/a/12204428/13697228
        move_row = self.val_df[self.val_df["index"] == next_index]
        self.train_df = self.train_df.append(move_row, ignore_index=True)
        self.val_df = self.val_df[self.val_df["index"] != next_index]
        # self.val_df = self.val_df.drop(index=next_index)

        next_experiment[proxy_name] = next_proxy
        next_experiment["score"] = next_score

        if print_experiment:
            print(pd.Series(next_experiment).to_frame().T)

        return next_experiment

    def closed_loop_adaptive_design(
        self,
        n_experiments=900,
        extraordinary_thresh=None,
        extraordinary_quantile=0.98,
        **suggest_next_experiment_kwargs,
    ):
        init_train_df = self.train_df
        if extraordinary_thresh is None:
            extraordinary_thresh = np.quantile(
                self.train_df.append(self.val_df).target.sort_values(),
                extraordinary_quantile,
            )
        self.extraordinary_thresh = extraordinary_thresh
        experiments = []
        first_experiment = self.suggest_first_experiment(
            **suggest_next_experiment_kwargs
        )
        experiments.append(first_experiment)
        for _ in tqdm(range(1, n_experiments)):
            next_experiment = self.suggest_next_experiment(
                **suggest_next_experiment_kwargs
            )
            experiments.append(next_experiment)

        experiment_df = ad_experiments_metrics(
            experiments, init_train_df, self.extraordinary_thresh
        )

        return experiment_df


def ad_experiments_metrics(experiments, train_df, extraordinary_thresh):
    experiment_df = pd.DataFrame(experiments)
    cummax, cumthresh, n_unique_atoms, n_unique_templates = ad_metrics(
        experiments, train_df, extraordinary_thresh
    )
    experiment_df["cummax"] = cummax
    experiment_df["cumthresh"] = cumthresh
    experiment_df["n_unique_atoms"] = n_unique_atoms
    experiment_df["n_unique_templates"] = n_unique_templates

    return experiment_df


def ad_metrics(experiments, init_train_df, extraordinary_thresh):
    init_train_formula = init_train_df.formula
    init_train_target = init_train_df.target
    init_max = max(init_train_target)
    experiment_df = pd.DataFrame(experiments)
    cummax = experiment_df.target.cummax()
    cummax[cummax <= init_max] = init_max
    # experiment_df.loc[experiment_df["cummax"] <= init_max, "cummax"] = init_max

    cumthresh = (experiment_df.target >= extraordinary_thresh).cumsum()

    atoms_list = set()
    templates = set()
    for formula in init_train_formula:
        atoms, _ = _fractional_composition_L(formula)
        _, counts = _element_composition_L(formula)
        atoms_list.update(atoms)
        counts = (
            np.array(counts).astype(int) / np.gcd.reduce(np.array(counts).astype(int))
        ).tolist()
        template = tuple(sorted(counts))
        templates.add(template)

    n_unique_atoms = []
    n_unique_templates = []

    for formula in experiment_df.formula:
        atoms, _ = _fractional_composition_L(formula)
        _, counts = _element_composition_L(formula)
        atoms_list.update(atoms)
        n_unique_atoms.append(len(atoms_list))
        counts = (
            np.array(counts).astype(int) / np.gcd.reduce(np.array(counts).astype(int))
        ).tolist()
        template = tuple(sorted(counts))
        templates.add(template)
        n_unique_templates.append(len(templates))

    return cummax, cumthresh, n_unique_atoms, n_unique_templates


# %% Code Graveyard
# if train_df is None and fit_afresh:
#     raise ValueError("if fit_afresh is True, train_df cannot be None.")
# if val_df is None and predict_afresh:
#     raise ValueError("if predict_afresh is True, val_df cannot be None.")
# if train_df is None or val_df is None and move_val_to_train:
#     raise ValueError(
#         "if move_val_to_train is True, both train_df and val_df must be supplied."
#     )
# if train_df is None:
#     use_self_train_df = True
#     train_df = self.train_df
#     if train_df is None:
#         raise ValueError(
#             "Run `disc.fit(train_df)` or supply `train_df` directly."
#         )
# else:
#     use_self_train_df = False
# if val_df is None:
#     use_self_val_df = True
#     val_df = self.val_df
#     if val_df is None:
#         raise ValueError(
#             "Run `disc.predict(val_df)` or supply `val_df` directly."
#         )
# else:
#     use_self_val_df = False

# if (use_self_train_df != use_self_val_df) and move_val_to_train:
#     if return_df:
#         warn(
#             "Manipulated DataFrames will be returned along with `next_experiment` as well as be manipulated internally (`disc.train_df` and `disc.val_df`)"
#         )
#     else:
#         warn(
#             "DataFrames will be manipulated internally (`disc.train_df` and `disc.val_df`) and only `next_experiment` will be returned."
#         )

# n_unique_atoms.append(len(atoms_list))
# n_unique_templates.append(len(templates))

# init_max = max(init_train_target)
# experiment_df = pd.DataFrame(experiments)
# experiment_df["cummax"] = experiment_df.target.cummax()
# experiment_df.loc[experiment_df["cummax"] <= init_max, "cummax"] = init_max

# experiment_df["cumthresh"] = (
#     experiment_df.target >= extraordinary_thresh
# ).cumsum()

# atoms_list = set()
# templates = set()
# for formula in init_train_formula:
#     atoms, _ = _fractional_composition_L(formula)
#     _, counts = _element_composition_L(formula)
#     atoms_list.update(atoms)
#     counts = (
#         np.array(counts).astype(int)
#         / np.gcd.reduce(np.array(counts).astype(int))
#     ).tolist()
#     template = tuple(sorted(counts))
#     templates.add(template)

# n_unique_atoms = []
# n_unique_templates = []

# for formula in experiment_df.formula:
#     atoms, _ = _fractional_composition_L(formula)
#     _, counts = _element_composition_L(formula)
#     atoms_list.update(atoms)
#     n_unique_atoms.append(len(atoms_list))
#     counts = (
#         np.array(counts).astype(int)
#         / np.gcd.reduce(np.array(counts).astype(int))
#     ).tolist()
#     template = tuple(sorted(counts))
#     templates.add(template)
#     n_unique_templates.append(len(templates))

# experiment_df["n_unique_atoms"] = n_unique_atoms
# experiment_df["n_unique_templates"] = n_unique_templates

# init_train_formula = self.train_df.formula
# init_train_target = self.train_df.target

