from tqdm import tqdm
import numpy as np
import pandas as pd
from crabnet.utils.composition import _fractional_composition_L, _element_composition_L
from mat_discover.mat_discover_ import Discover, my_mvn


class Adapt(Discover):
    def __init__(self, train_df, val_df, **Discover_kwargs):
        super().__init__(**Discover_kwargs)
        self.train_df = train_df
        self.val_df = val_df

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
                val_true, self.val_pred, _, val_sigma = self.crabnet_model.predict(
                    self.val_df
                )
                # convert back to NumPy arrays
                train_emb = np.array(self.train_df.emb.tolist())
                val_emb = np.array(self.val_df.emb.tolist())

                train_r_orig = self.train_df.r_orig.values
                mvn_list = list(
                    map(my_mvn, train_emb[:, 0], train_emb[:, 1], train_r_orig)
                )
                pdf_list = [mvn.pdf(val_emb) for mvn in mvn_list]
                self.val_dens = np.sum(pdf_list, axis=0)
                self.val_log_dens = np.log(self.val_dens)
                # recompute dens score
                self.dens_score = self.weighted_score(
                    self.val_pred,
                    self.val_dens,
                    pred_weight=self.pred_weight,
                    proxy_weight=self.proxy_weight,
                )
                self.dens_score_df = self.sort(self.dens_score)

            proxy_lookup = {
                "density": "dens_score_df",
                "peak": "peak_score_df",
                "radius": "rad_score_df",
            }
            proxy_df_name = proxy_lookup[proxy_name]
            proxy_df = getattr(self, proxy_df_name)
            next_formula, next_target, next_proxy, next_score = [
                proxy_df[name].iloc[0]
                for name in ["formula", "prediction", proxy_name, "score"]
            ]
            next_index, next_emb, next_r_orig = (
                self.val_df[self.val_df.formula == next_formula]
                .dropna()[["index", "emb", "r_orig"]]
                .iloc[0]
            )
        else:
            sample = self.val_df.sample(1)
            next_formula, next_target, next_index = sample[
                ["formula", "target", "index"]
            ].values[0]
            next_proxy = np.nan
            next_score = np.nan
            next_emb = np.nan
            next_r_orig = np.nan

        next_experiment = {
            "formula": next_formula,
            "index": next_index,
            "target": next_target,
            "emb": next_emb,
            "r_orig": next_r_orig,
        }

        # append compound to train, remove from val
        self.train_df = self.train_df.append(next_experiment, ignore_index=True)
        self.val_df = self.val_df[self.val_df["index"] != next_experiment["index"]]

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
        if extraordinary_thresh is None:
            extraordinary_thresh = np.quantile(
                self.train_df.append(self.val_df).target.sort_values(),
                extraordinary_quantile,
            )
        self.extraordinary_thresh = extraordinary_thresh
        experiments = []
        for i in tqdm(range(n_experiments)):
            if i == 0:
                next_experiment = self.suggest_first_experiment(
                    **suggest_next_experiment_kwargs
                )
            else:
                next_experiment = self.suggest_next_experiment(
                    **suggest_next_experiment_kwargs
                )
            experiments.append(next_experiment)

        experiment_df = pd.DataFrame(experiments)
        experiment_df["cummax"] = experiment_df.target.cummax()

        experiment_df["cumthresh"] = (
            experiment_df.target >= extraordinary_thresh
        ).cumsum()

        n_unique_atoms = []
        atoms_list = set()
        n_unique_templates = []
        templates = set()
        for formula in experiment_df.formula:
            atoms, _ = _fractional_composition_L(formula)
            _, counts = _element_composition_L(formula)
            atoms_list.update(atoms)
            n_unique_atoms.append(len(atoms_list))
            counts = (
                np.array(counts).astype(int)
                / np.gcd.reduce(np.array(counts).astype(int))
            ).tolist()
            template = tuple(sorted(counts))
            templates.add(template)
            n_unique_templates.append(len(templates))

        experiment_df["n_unique_atoms"] = n_unique_atoms
        experiment_df["n_unique_templates"] = n_unique_templates

        return experiment_df


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

