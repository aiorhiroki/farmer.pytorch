import train_imp


def command():
    train_anno, val_anno = train_imp.GetAnnotationImp()()  # アノテーションファイル取得

    model = train_imp.get_model_task()  # モデル構築
    loss_func = train_imp.get_loss_func_task()  # 損失関数
    metrics = train_imp.get_metrics_task()  # 評価指標
    augmentation = train_imp.get_augmentation_task()  # データ拡張方法の定義
    # データ読み込み・前処理
    train_dataset = train_imp.DatasetImp(train_anno, augmentation)
    val_dataset = train_imp.DatasetImp(val_anno, augmentation)
    # 学習
    train_imp.GetOptimizationImp(
        model, loss_func, metrics, train_dataset, val_dataset)()


if __name__ == "__main__":
    command()
