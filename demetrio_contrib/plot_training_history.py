from stable_baselines3.common.monitor import load_results


def load_results_df(log_dir):
    df_res = load_results(log_dir)
    df_res['steps_trained'] = df_res.l.cumsum()
    df_res.rename(columns={'r': 'reward'}, inplace=True)
    return df_res


def plot_training_history(
    log_dir,
    save_path=None,
    rolling_avg_window=0,

):
    df_res = load_results_df(log_dir)
    if rolling_avg_window:
        ra_col = f'reward_RA_{rolling_avg_window}'
        df_res[ra_col] = df_res.rolling(window=rolling_avg_window)['reward'].mean()


    import seaborn as sns
    sns_plt = sns.lineplot(x='steps_trained', y='reward', alpha=0.5, data=df_res)
    if rolling_avg_window:
        sns.lineplot(x='steps_trained', y=ra_col, alpha=0.7, linestyle=':', data=df_res, ax=sns_plt)
    if save_path is not None:
        sns_plt.figure.savefig(save_path)
    return sns_plt