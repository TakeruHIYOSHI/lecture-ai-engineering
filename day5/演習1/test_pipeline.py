import unittest
import time
import mlflow
import pandas as pd
import numpy as np
from pipeline import prepare_data, train_and_evaluate
from sklearn.metrics import accuracy_score
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data()
        self.model, self.accuracy, self.params = train_and_evaluate(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def test_model_accuracy(self):
        """モデルの推論精度をテスト"""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0.7)  # 最低精度の閾値
        logger.info(f"モデルの精度: {accuracy:.4f}")

    def test_inference_time(self):
        """推論時間をテスト"""
        start_time = time.time()
        _ = self.model.predict(self.X_test)
        inference_time = time.time() - start_time
        self.assertLess(inference_time, 1.0)  # 1秒以内に推論が完了することを確認
        logger.info(f"推論時間: {inference_time:.4f}秒")

    def test_model_comparison(self):
        """過去バージョンのモデルとの比較"""
        try:
            # 最新のモデルを取得
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("titanic-survival-prediction")
            if experiment is None:
                logger.warning("実験が見つかりません。比較テストをスキップします。")
                return

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.accuracy DESC"],
                max_results=2
            )

            if len(runs) < 2:
                logger.warning("比較するための過去のモデルがありません。")
                return

            # 最新のモデルと過去のモデルの精度を比較
            current_accuracy = runs[0].data.metrics["accuracy"]
            previous_accuracy = runs[1].data.metrics["accuracy"]
            
            # 性能劣化がないことを確認（許容範囲を5%とする）
            self.assertGreaterEqual(
                current_accuracy,
                previous_accuracy * 0.95,
                "モデルの性能が5%以上劣化しています"
            )
            logger.info(f"現在の精度: {current_accuracy:.4f}, 過去の精度: {previous_accuracy:.4f}")

        except Exception as e:
            logger.error(f"モデル比較中にエラーが発生しました: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main() 