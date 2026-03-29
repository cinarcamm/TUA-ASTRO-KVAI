import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AIAnalyst:

    def __init__(
        self,
        window_size: int = 16,
        std_threshold: float = 1e-6,
        contamination: float = 0.01,
        random_state: int = 42,
        n_estimators: int = 50,
    ) -> None:
       
        if window_size < 2:
            raise ValueError("window_size en az 2 olmalıdır.")
        if not (0 < contamination < 0.5):
            raise ValueError("contamination 0 ile 0.5 arasında olmalıdır.")

        self.window_size = window_size
        self.std_threshold = std_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators

    @staticmethod
    def _to_1d_array(data: np.ndarray | pd.Series) -> np.ndarray:
        """
        Girdiyi 1D numpy dizisine dönüştürür.
        """
        if isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=float)
        elif isinstance(data, np.ndarray):
            arr = data.astype(float, copy=False)
        else:
            raise TypeError("Girdi numpy.ndarray veya pandas.Series olmalıdır.")

        if arr.ndim != 1:
            raise ValueError("Girdi tek boyutlu (1D) olmalıdır.")
        if arr.size == 0:
            raise ValueError("Girdi boş olamaz.")
        if not np.isfinite(arr).all():
            raise ValueError("Girdi NaN/Inf içermemelidir.")

        return arr

    def detect_stuck_at_faults(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """
        Rolling window std yaklaşımı ile donmuş blokları tespit eder.

        Dönüş:
            stuck_mask: bool ndarray, True olan indeksler stuck-at fault kabul edilir.
        """
        x = self._to_1d_array(data)
        n = x.size
        w = self.window_size

        if n < w:
            return np.zeros(n, dtype=bool)

        # O(N) rolling variance hesabı için kümülatif toplamlar
        csum = np.concatenate(([0.0], np.cumsum(x)))
        csum2 = np.concatenate(([0.0], np.cumsum(x * x)))

        win_sum = csum[w:] - csum[:-w]
        win_sum2 = csum2[w:] - csum2[:-w]

        mean = win_sum / w
        var = np.maximum((win_sum2 / w) - (mean * mean), 0.0)
        std = np.sqrt(var)

        low_var_windows = std <= self.std_threshold  # pencere bazlı

        # Pencere tespitini örnek bazlı maskeye yay
        stuck_mask = np.zeros(n, dtype=bool)
        # Bu döngü pratikte hızlıdır; her pencere en fazla w eleman işaretler.
        # window_size sabit olduğunda toplam maliyet O(N) olur.
        idxs = np.where(low_var_windows)[0]
        for i in idxs:
            stuck_mask[i : i + w] = True

        return stuck_mask

    def detect_seu_anomalies(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """
        IsolationForest ile sinsi kozmik radyasyon kaynaklı anomali (SEU) tespiti.

        Dönüş:
            seu_mask: bool ndarray, True olan indeksler anomali kabul edilir.
        """
        x = self._to_1d_array(data)

        # IsolationForest 2D giriş ister: (n_samples, n_features)
        X = x.reshape(-1, 1)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,  # edge cihaz uyumluluğu ve kaynak kontrolü
        )
        pred = model.fit_predict(X)  # normal: 1, anomali: -1
        seu_mask = pred == -1
        return seu_mask

    def generate_ai_report(self, stuck_mask: np.ndarray, seu_mask: np.ndarray) -> str:
        """
        Tespit sonuçlarını teknik bir rapor metnine dönüştürür.
        """
        if stuck_mask.dtype != bool or seu_mask.dtype != bool:
            raise TypeError("Maskeler bool türünde olmalıdır.")
        if stuck_mask.shape != seu_mask.shape:
            raise ValueError("Maskeler aynı boyutta olmalıdır.")

        stuck_count = int(np.sum(stuck_mask))
        seu_count = int(np.sum(seu_mask))

        # Donma blok sayısını yaklaşık segment sayısı olarak raporla
        # (ardışık True bölgeleri tek blok kabul edilir)
        transitions = np.diff(stuck_mask.astype(np.int8), prepend=0)
        stuck_block_count = int(np.sum(transitions == 1))

        report = (
            "AI Analiz Raporu:\n"
            f"- Stuck-at Fault: {stuck_block_count} adet donma bloğu, toplam {stuck_count} örnek etkilendi.\n"
            f"- SEU/Anomali: {seu_count} adet örnek IsolationForest tarafından anomali olarak işaretlendi.\n"
            "- Yöntem Notu: Donma tespiti rolling std tabanlı, SEU tespiti IsolationForest tabanlıdır.\n"
            "- Karmaşıklık: Donma tespiti O(N), anomali maskeleme veri geçişi O(N) olacak şekilde tasarlanmıştır."
        )
        return report

    def analyze(self, data: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Uçtan uca analiz:
        1) Stuck-at fault maskesi
        2) SEU maskesi
        3) Teknik rapor metni

        Dönüş:
            (stuck_mask, seu_mask, report)
        """
        stuck_mask = self.detect_stuck_at_faults(data)
        seu_mask = self.detect_seu_anomalies(data)
        report = self.generate_ai_report(stuck_mask, seu_mask)
        return stuck_mask, seu_mask, report
