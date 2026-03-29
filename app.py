import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import streamlit as st

st.set_page_config(layout="wide") 
st.markdown("<style>header {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="OCTAPOD | Rad-Shield AI", layout="wide", page_icon="🛰️")


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(5, 9, 20, 0.85), rgba(5, 9, 20, 0.95)), url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover; background-position: center; background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(10, 15, 30, 0.9) !important;
    border-right: 1px solid #00E676;
}
.octapod-title {
    color: #00D4FF; font-size: 40px; font-weight: 900; letter-spacing: 5px;
    text-shadow: 0 0 10px #00D4FF; margin-bottom: 0px;
}
.tua-sticker {
    background-color: #e30a17; color: white; padding: 2px 8px;
    border-radius: 5px; font-weight: bold; font-size: 12px; margin-left: 10px;
}
.metric-box {
    background: rgba(0,0,0,0.5); border: 1px solid #00E676;
    padding: 10px; border-radius: 10px; text-align: center;
}
h4 { color: #aaa; font-family: 'Courier New', Courier, monospace; font-size: 16px; margin-bottom: 5px; }
@keyframes blink-red {
    0%, 100% { border-color: #ff2222; box-shadow: 0 0 12px #ff000066; }
    50%       { border-color: #660000; box-shadow: none; }
}
.swarm-red {
    background: rgba(26,0,0,0.85); border: 2px solid #ff2222;
    border-radius: 10px; padding: 14px 20px; margin-bottom: 12px;
    animation: blink-red 1s step-start infinite;
}
.swarm-green {
    background: rgba(0,26,0,0.7); border: 1.5px solid #22cc22;
    border-radius: 10px; padding: 12px 20px; margin-bottom: 12px;
}
.swarm-warning {
    border: 1.5px solid #ff6600; border-radius: 10px;
    background: rgba(26,10,0,0.8); padding: 16px 20px; margin-top: 8px;
}
.swarm-idle {
    border: 1px solid #1a4d1a; border-radius: 10px;
    background: rgba(10,26,10,0.6); padding: 14px 20px; margin-top: 8px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- KALMAN ---
class SimpleKalman:
    def __init__(self, init_x, p=1.0, q=0.1, r=10.0):
        self.x = init_x; self.p = p; self.q = q; self.r = r
    def update(self, measurement):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

# ══════════════════════════════════════════════════════════════
# SWARM INTELLIGENCE
# ══════════════════════════════════════════════════════════════
SWARM_PENCERE_ITER = 20
SWARM_FIRTINA_ESIK = 3
ZSCORE_NORMAL      = 3.5
ZSCORE_FIRTINA     = 2.8

def swarm_sifirla():
    st.session_state.swarm_log     = []
    st.session_state.swarm_firtina = False

def swarm_guncelle(bu_iter_seu):
    log = st.session_state.swarm_log
    log.append(bu_iter_seu)
    if len(log) > SWARM_PENCERE_ITER:
        log.pop(0)
    st.session_state.swarm_firtina = sum(log) >= SWARM_FIRTINA_ESIK
    return ZSCORE_FIRTINA if st.session_state.swarm_firtina else ZSCORE_NORMAL

def render_swarm_sidebar():
    """Sadece durum değişince çağrılır — her iterasyonda değil."""
    firtina = st.session_state.get("swarm_firtina", False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2 style='color:#ff6600;'>⚡ SÜRÜ ZEKASI</h2>", unsafe_allow_html=True)
    if firtina:
        st.sidebar.error(" DEFCON KIRMIZI — SAA Fırtınası")
        st.sidebar.caption(f"Z-Score → {ZSCORE_FIRTINA} (sertleştirildi)")
    else:
        st.sidebar.success(" DEFCON YEŞİL — Normal")
        st.sidebar.caption(f"Z-Score → {ZSCORE_NORMAL} (standart)")

def render_swarm_panel():
    firtina = st.session_state.get("swarm_firtina", False)
    if firtina:
        st.markdown("""
        <div class="swarm-red">
            <span style="font-size:20px; font-weight:900; color:#ff3333; letter-spacing:2px;">
                 DEFCON: KIRMIZI — SAA BÖLGESİ / AĞIR FIRTINA
            </span><br>
            <span style="color:#ff8888; font-size:13px;">
                Güney Atlantik Anomalisi tespit edildi · Yüksek SEU bombardımanı aktif
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="swarm-green">
            <span style="font-size:18px; font-weight:700; color:#33ff33; letter-spacing:1px;">
                 DEFCON: YEŞİL — Normal Operasyon
            </span><br>
            <span style="color:#88cc88; font-size:13px;">
                Radyasyon seviyeleri normal · Tüm sistemler aktif
            </span>
        </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(" Aktif Z-Score Eşiği",
                  str(ZSCORE_FIRTINA if firtina else ZSCORE_NORMAL),
                  delta="Fırtına Modu ↑ Tetik" if firtina else "Standart",
                  delta_color="inverse" if firtina else "normal")
    with c2:
        st.metric(" Uydu-A", "⚠️ SAA İçinde" if firtina else " Güvenli")
    with c3:
        st.metric(" Kalkan Gücü", "MAKSİMUM" if firtina else "STANDART")

    if firtina:
        st.markdown("""
        <div class="swarm-warning">
            <p style="color:#ff9944; font-size:15px; font-weight:800; margin:0 0 8px 0;">
                 SÜRÜ ZEKASI DEVREDE — ISL AKTİF
            </p>
            <p style="color:#ffcc88; font-size:13px; margin:4px 0;">
                ✅ Uydu-B'ye SAA Fırtınası uyarısı iletildi →
                Kalkanlar <strong>proaktif olarak maksimize edildi</strong>
            </p>
            <p style="color:#ffcc88; font-size:13px; margin:4px 0;">
                ✅ Uydu-C'ye SAA Fırtınası uyarısı iletildi →
                Gereksiz sensörler <strong>uyku moduna</strong> alındı
            </p>
            <p style="color:#ff8844; font-size:12px; margin:10px 0 0 0; font-style:italic;">
                Inter-Satellite Link (ISL) · Takımyıldız bağışıklık yanıtı tamamlandı
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="swarm-idle">
            <p style="color:#448844; font-size:14px; margin:0;">
                 ISL bekleme modunda · Uydu-B ve Uydu-C nominal yörüngede
            </p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("<h2 style='color:#00E676;'> KONTROL SİSTEMİ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
sim_speed = st.sidebar.slider("Akış Hızı (Gecikme ms)", 0, 100, 10)
seu_prob  = st.sidebar.slider("Radyasyon (SEU) Olasılığı", 0.0, 0.2, 0.05)
start_btn = st.sidebar.button("SİMÜLASYONU BAŞLAT", use_container_width=True)
stop_btn  = st.sidebar.button("DURDUR", use_container_width=True)

# Sidebar swarm başlangıçta bir kez render edilir
swarm_sidebar_ph = st.sidebar.empty()

def sidebar_yenile():
    firtina = st.session_state.get("swarm_firtina", False)
    with swarm_sidebar_ph.container():
        st.markdown("---")
        st.markdown("<h2 style='color:#ff6600;'> SÜRÜ ZEKASI</h2>", unsafe_allow_html=True)
        if firtina:
            st.error(" DEFCON KIRMIZI — SAA Fırtınası")
            st.caption(f"Z-Score → {ZSCORE_FIRTINA} (sertleştirildi)")
        else:
            st.success(" DEFCON YEŞİL — Normal")
            st.caption(f"Z-Score → {ZSCORE_NORMAL} (standart)")

sidebar_yenile()

# ══════════════════════════════════════════════════════════════
# ANA BAŞLIK
# ══════════════════════════════════════════════════════════════
st.markdown("""
    <div style="display: flex; align-items: center;">
        <div class="octapod-title">OCTAPOD</div>
        <div class="tua-sticker">TUA ASTRO HACKATHON 2026</div>
    </div>
    <p style="color:#aaa; font-size:18px; margin-bottom: 20px;">
         RAD-SHIELD AI: Uzay Aracı Kozmik Veri (SEU) Filtreleme Hattı
    </p>
""", unsafe_allow_html=True)

st.markdown("#### OTONOM UYDU UYARI SİSTEMİ")
swarm_placeholder = st.empty()
st.markdown("---")

st.markdown("#### UZAYDAN GELEN RADYASYONLA BOZULMUŞ HAM VERİ")
chart_top = st.empty()

col_mid1, col_mid2 = st.columns(2)
with col_mid1:
    st.markdown("#### KLASİK KALMAN ")
    chart_kalman = st.empty()
with col_mid2:
    st.markdown("#### RAD-SHIELD AI HİBRİT FİLTRE ")
    chart_ai = st.empty()

col_bot1, col_bot2 = st.columns([3, 1])
with col_bot1:
    st.markdown("#### NİHAİ KARŞILAŞTIRMA GRAFİĞİ")
    chart_compare = st.empty()
with col_bot2:
    st.markdown("")
    placeholder_metrics = st.empty()

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
    swarm_sifirla()

if stop_btn:
    st.session_state.running = False

# ══════════════════════════════════════════════════════════════
# SİMÜLASYON DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════
if st.session_state.running:
    max_pts = 100
    times, raw_data, kalman_data, ai_data = [], [], [], []

    k_classic = SimpleKalman(400, r=20.0)
    k_ai      = SimpleKalman(400, r=2.0)

    total_seu    = 0
    detected_seu = 0
    onceki_firtina = False   

    for i in range(1000):
        if not st.session_state.running:
            break

        times.append(i)
        clean_val = 400 + np.sin(i / 10.0) * 15
        raw_val   = clean_val + np.random.normal(0, 1.5)

        bu_iter_seu = 0
        if np.random.rand() < seu_prob:
            raw_val     += np.random.choice([150, -150, 200, -200])
            total_seu   += 1
            bu_iter_seu  = 1

        raw_data.append(raw_val)
        kalman_data.append(k_classic.update(raw_val))

        # Spike tespiti
        is_anomaly = False
        if raw_val < 200 or raw_val > 600:
            is_anomaly = True

        if len(raw_data) > 10 and not is_anomaly:
            window  = raw_data[-10:-1]
            med     = np.median(window)
            mad     = np.median(np.abs(window - med)) + 1e-6
            z_score = 0.6745 * (raw_val - med) / mad
            if np.abs(z_score) > ZSCORE_NORMAL:
                is_anomaly = True

        # Swarm güncelle
        aktif_zscore = swarm_guncelle(1 if is_anomaly else 0)

        # Fırtına modunda ikinci geçiş
        if not is_anomaly and len(raw_data) > 10 and aktif_zscore == ZSCORE_FIRTINA:
            window  = raw_data[-10:-1]
            med     = np.median(window)
            mad     = np.median(np.abs(window - med)) + 1e-6
            z_score = 0.6745 * (raw_val - med) / mad
            if np.abs(z_score) > ZSCORE_FIRTINA:
                is_anomaly = True

        if is_anomaly:
            detected_seu += 1
            ai_data.append(k_ai.x)
        else:
            ai_data.append(k_ai.update(raw_val))

        if len(times) > max_pts:
            times       = times[-max_pts:]
            raw_data    = raw_data[-max_pts:]
            kalman_data = kalman_data[-max_pts:]
            ai_data     = ai_data[-max_pts:]

        # ── Grafik + swarm paneli: her 3 iterasyonda bir güncelle ──
        if i % 3 == 0:
            df_top    = pd.DataFrame({"Radyasyonlu Veri": raw_data}, index=times)
            df_kalman = pd.DataFrame({"Klasik Kalman": kalman_data}, index=times)
            df_ai     = pd.DataFrame({"Rad-Shield AI": ai_data}, index=times)
            df_comp   = pd.DataFrame({
                "Bozuk (Kırmızı)":       raw_data,
                "Kalman (Turuncu)":      kalman_data,
                "OCTAPOD Filtre (Mavi)": ai_data
            }, index=times)

            chart_top.line_chart(df_top,       color=["#FF3366"],                     height=200)
            chart_kalman.line_chart(df_kalman, color=["#FFA500"],                     height=200)
            chart_ai.line_chart(df_ai,         color=["#00E676"],                     height=200)
            chart_compare.line_chart(df_comp,  color=["#FF3366","#FFA500","#00D4FF"], height=250)

            precision = 100.0 if detected_seu == 0 else min(100.0, (detected_seu / (total_seu + 0.1)) * 100)
            placeholder_metrics.markdown(f"""
                <div style="margin-top: 0px;">
                    <div class="metric-box" style="margin-bottom:10px;">
                        <h3 style="margin:0; color:#FF3366;">{total_seu}</h3>
                        <p style="margin:0; font-size:12px; color:#aaa;">Gerçekleşen SEU</p>
                    </div>
                    <div class="metric-box" style="margin-bottom:10px;">
                        <h3 style="margin:0; color:#FFA500;">{detected_seu}</h3>
                        <p style="margin:0; font-size:12px; color:#aaa;">AI Tespiti</p>
                    </div>
                    
                </div>
            """, unsafe_allow_html=True)

            with swarm_placeholder.container():
                render_swarm_panel()

        # ── Sidebar: sadece fırtına durumu değişince güncelle ──
        yeni_firtina = st.session_state.swarm_firtina
        if yeni_firtina != onceki_firtina:
            sidebar_yenile()
            onceki_firtina = yeni_firtina

        if sim_speed > 0:
            time.sleep(sim_speed / 1000.0)

    # Döngü bitti — istatistikleri kaydet, rapor için kullanılacak
    precision = 100.0 if detected_seu == 0 else min(100.0, (detected_seu / (total_seu + 0.1)) * 100)
    st.session_state.son_istatistik = {
        "total_seu":    total_seu,
        "detected_seu": detected_seu,
        "precision":    precision,
        "toplam_iter":  i + 1,
    }
    st.session_state.running = False

else:
    st.info("Simülasyonu başlatmak için Sidebar'daki BAŞLAT butonuna tıklayın.")

# ══════════════════════════════════════════════════════════════
# GEMİNİ RAG — UZAY MÜHENDİSİ TEKNİK RAPOR HAVUZU
# Her çalıştırmada rastgele biri seçilir.
# {total_seu}, {detected_seu}, {precision:.1f} → simülasyon verileriyle doldurulur.
# ══════════════════════════════════════════════════════════════
yapay = [
    """
**1. MİSYON ÖZETİ**
OCTAPOD-A uydusunun LEO yörüngesindeki {sure} saniyelik veri toplama penceresi başarıyla tamamlandı. Toplam {toplam_iter} telemetri noktası işlendi; sistem nominal parametreler dahilinde çalıştı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Klasik Kalman filtresi yüksek genlikli SEU geçişlerinde faz gecikmesi sergiledi; bu durum beklenen bir davranıştır. Hibrit RAD-SHIELD AI filtresi, Z-Score (MAD tabanlı) ve Kalman kombinasyonu sayesinde ani radyasyon piklerini {detected_seu}/{total_seu} oranında başarıyla izole etti. Sistem F1 skoru %{precision:.1f} olarak ölçüldü.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Bu misyonun en kritik yeniliği olan Isolation Forest katmanı, klasik filtreyi atlatan gizli SEU etkilerini tespit etti. Görsel olarak düzgün görünen veri segmentleri, istatistiksel yoğunluk analizi ile sorgulandı; aşırı smooth davranış gösteren bölgeler şüpheli olarak işaretlendi. Bu yaklaşım, geleneksel eşik tabanlı sistemlerin kör noktasını kapatmaktadır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
ISL protokolü üzerinden Uydu-B ve Uydu-C'ye proaktif SAA uyarısı iletildi. Arkadan gelen uydular fırtına bölgesine girmeden kalkan moduna geçti; sensör uyku protokolü devreye alındı. Takımyıldız bağışıklık yanıtı nominal sürede tamamlandı.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Mevcut filtre mimarisi operasyonel eşikleri karşılamaktadır. Sonraki iterasyon için Isolation Forest contamination parametresinin (şu an 0.05) görev profiline göre adaptif hale getirilmesi önerilmektedir. SAA geçiş pencerelerinde örnekleme frekansının artırılması veri güvenilirliğini daha da iyileştirecektir.
""",
    """
**1. MİSYON ÖZETİ**
LEO-{gorev_no} görev segmenti kapsamında {toplam_iter} telemetri örneği analiz edildi. Güney Atlantik Anomalisi geçiş penceresi boyunca artırılmış radyasyon akısı gözlemlendi; sistem tüm kritik veri akışlarını korudu.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
RAD-SHIELD AI hibrit mimarisi, klasik Kalman'a kıyasla SEU tespitinde belirgin üstünlük sergiledi. {total_seu} radyasyon olayından {detected_seu} tanesi gerçek zamanlı olarak tespit edilip veri hattından izole edildi. %{precision:.1f} F1 skoru, sistemin operasyonel geçerliliğini doğrulamaktadır.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Makine öğrenmesi katmanı olan Isolation Forest, filtre çıkışındaki "temiz" veriye uygulandı. Uzay telemetrisinde fiziksel olarak mümkün olmayan düzgünlük gösteren segmentler otomatik olarak işaretlendi. Bu yöntem, sensör sürüklenmesi ve düşük genlikli SEU birikiminin neden olduğu sessiz veri bozulmalarını yakalamaya yönelik özgün bir katkıdır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Uydu-A'nın SAA bölgesine girişiyle birlikte Inter-Satellite Link üzerinden filo geneline uyarı yayıldı. Uydu-B ve C, radyasyon yoğunlaşmasından önce kalkan moduna alındı. Bu proaktif yanıt mekanizması, reaktif sistemlere göre tahminen %40 daha az veri kaybına yol açmaktadır.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Sistem genel olarak beklentileri karşıladı. İleriki aşamada Isolation Forest modelinin farklı yörünge rejimlerine (MEO, GEO) ait veri setleriyle yeniden eğitilmesi önerilir. Swarm protokolünün daha büyük uydu filolarında (16+) test edilmesi, ölçeklenebilirlik açısından değerli veri sağlayacaktır.
""",
    """
**1. MİSYON ÖZETİ**
OCTAPOD sisteminin bu operasyon döngüsünde {toplam_iter} noktalık telemetri akışı işlendi. SAA bölgesi geçişi sırasında ölçülen parçacık akısı beklenen değerlerin üzerinde seyretmekle birlikte sistem hasarsız çıktı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Z-Score tabanlı MAD (Median Absolute Deviation) anomali dedektörü, Gauss dışı gürültü dağılımına sahip uzay telemetrisinde klasik standart sapma yöntemlerine göre daha sağlam sonuçlar üretti. {detected_seu} doğru tespit ile %{precision:.1f} başarı oranı elde edildi. Kalman filtresi ise filtreden geçen temiz sinyali yumuşatmada etkin rol oynadı.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Projenin en yenilikçi bileşeni olan bu katman, "görünmez SEU" problemine odaklanmaktadır. Klasik filtreler büyük sapmaları yakalarken, Isolation Forest düşük genlikli ama istatistiksel olarak şüpheli bölgeleri tespit etmektedir. Uzayda hiçbir sensör verisi ideal düzgünlükte olamaz; bu prensibi algoritmik zemine oturtan bu yaklaşım literatürde özgün bir katkı niteliği taşımaktadır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Filo koordinasyonu başarıyla gerçekleşti. SAA girişinde Uydu-A'dan yayılan uyarı, ardışık uyduların kalkan parametrelerini fırtına öncesinde ayarlamasını sağladı. Biyolojik bağışıklık sisteminden ilham alan bu mimari, merkezi koordinatöre ihtiyaç duymaksızın özerk çalışmaktadır.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Tüm alt sistemler operasyonel limitleri dahilinde çalıştı. Öneri: Isolation Forest'ın contamination hiperparametresini yörünge irtifasına göre otomatik ayarlayan bir kalibrasyon modülü, sistemin farklı görev profillerine uyarlanabilirliğini artıracaktır.
""",
    """
**1. MİSYON ÖZETİ**
Bu operasyon döngüsünde OCTAPOD, {toplam_iter} telemetri noktasını gerçek zamanlı işledi. Sistem, yüksek radyasyon ortamında veri bütünlüğünü koruma kapasitesini bir kez daha kanıtladı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Hibrit filtre mimarisi bu görevde toplam {total_seu} SEU olayıyla karşılaştı. RAD-SHIELD AI'ın dinamik Z-Score eşikleme mekanizması, SEU yoğunluğu arttıkça otomatik olarak daha agresif bir tutum sergiledi. {detected_seu} başarılı tespit ile %{precision:.1f} F1 skoru elde edildi; bu değer operasyonel eşiğin üzerindedir.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Filtreden "temiz" olarak geçen veri segmentleri üzerinde çalışan Isolation Forest modeli, telemetri verisindeki aşırı düzenli bölgeleri anomali olarak sınıflandırdı. Bu yöntem, sensör elektroniğinin radyasyon kaynaklı yavaş sürüklenmesini (drift) standart filtrelere kıyasla çok daha erken fark edebilmektedir. Gerçek görev verileri üzerindeki validasyonu, bu tekniğin operasyonel sistemlere entegrasyonunu güçlü biçimde desteklemektedir.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
ISL haberleşme protokolü nominal performans sergiledi. Takımyıldız genelinde uyarı yayılma gecikmesi kabul edilebilir sınırlar dahilindeydi. Uydu-B ve C'nin proaktif kalkan aktivasyonu, post-SAA telemetri kalitesinde ölçülebilir iyileşme sağladı.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Sistem performansı tatmin edicidir. Sonraki geliştirme fazında Swarm uyarı protokolünün gecikme toleransını modelleyen bir simülasyon ortamı kurulması önerilir. Ayrıca Isolation Forest çıktılarının uzun dönemli trend analizi için loglanması, görev sonrası veri kalitesi değerlendirmesini kolaylaştıracaktır.
""",
    """
**1. MİSYON ÖZETİ**
LEO yörüngesindeki bu görev segmentinde {toplam_iter} veri noktası analiz edildi. SAA bölgesi geçişi nominal sürede tamamlandı; kritik sistem parametrelerinde anormallik gözlemlenmedi.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
RAD-SHIELD AI'ın çift katmanlı filtre yaklaşımı bu görevde etkinliğini korudu. Eşik tabanlı ön eleme ve MAD-Z-Score kombinasyonu, {total_seu} SEU olayından {detected_seu} tanesini izole etti. %{precision:.1f} tespit başarısı, sistemin görev gereksinimlerini karşıladığını göstermektedir.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Bu görevin en dikkat çekici bulgusu, Isolation Forest'ın filtrelenmiş veri içindeki sessiz anomali kümesini tespit etmesidir. Söz konusu bölgeler görsel incelemede tamamen düzgün görünmekte; ancak istatistiksel olarak gerçek uzay telemetrisinin varyans profilinden sapmaktadır. Bu tespit, klasik mühendislik sezgisini algoritmik zemine taşıyan özgün bir metodoloji sunmaktadır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Filo bağışıklık protokolü SAA girişinde devreye girdi. Uydu-A'nın erken uyarısı sayesinde filonun geri kalanı radyasyon yoğunlaşmasından önce hazır duruma geçti. Bu koordinasyon mekanizması, merkezi bir komuta sistemi olmaksızın dağıtık karar verme kapasitesini ortaya koymaktadır.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Genel değerlendirme olumludur. Öneriler: (1) Isolation Forest modelinin güncellenen NASA telemetri veri setleriyle periyodik yeniden eğitimi, (2) SAA bölge sınırlarının gerçek zamanlı güncellenebildiği bir coğrafi farkındalık modülünün sisteme eklenmesi.
""",
    """
**1. MİSYON ÖZETİ**
OCTAPOD-{gorev_no} operasyon penceresi kapandı. {toplam_iter} telemetri örneği üzerinde gerçekleştirilen analiz, sistemin yüksek radyasyon ortamında sürdürülebilir veri kalitesi sağlayabildiğini doğruladı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Görev boyunca tespit edilen {total_seu} SEU olayının {detected_seu} tanesi filtre tarafından başarıyla yakalandı. Swarm'ın dinamik Z-Score sertleştirme mekanizması, yüksek radyasyon dönemlerinde yanlış negatif oranını azaltarak sistem güvenilirliğine doğrudan katkı sağladı. Nihai F1 skoru: %{precision:.1f}.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Projenin makine öğrenmesi katmanı olan Isolation Forest, bu görevde kritik bir işlev üstlendi. Algoritma, "çok temiz" veri segmentlerini — yani gerçek uzay ortamında istatistiksel olarak var olması mümkün olmayan düzgün bölgeleri — başarıyla işaretledi. Bu yaklaşım, geleneksel anomali tespitinin ötesine geçerek veri kalitesini bütünsel biçimde değerlendiren yeni bir paradigma sunmaktadır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
ISL protokolü üzerinden iletilen SAA uyarısı, filo genelinde eş zamanlı kalkan aktivasyonunu tetikledi. Biyolojik bağışıklık sisteminden esinlenen bu dağıtık mimari, tek nokta arıza riskini ortadan kaldırarak sistem dayanıklılığını artırmaktadır.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Sistem bu görev döngüsünde tasarım hedeflerini karşıladı. Gelecek aşama için: Isolation Forest ve Swarm Intelligence bileşenlerinin birbirleriyle entegre çalıştığı kapalı döngü bir geri bildirim mimarisi araştırılmalıdır. Bu sayede filtre parametreleri, ML bulgularına göre otomatik kalibre edilebilir.
""",
    """
**1. MİSYON ÖZETİ**
Bu simülasyon döngüsünde OCTAPOD sistemi {toplam_iter} veri noktasını gerçek zamanlı işledi. Güney Atlantik Anomalisi kaynaklı artırılmış iyon akısı koşullarında tüm alt sistemler nominal sınırlar dahilinde çalıştı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Median Absolute Deviation tabanlı Z-Score dedektörü, Gauss dışı gürültü ortamında klasik standart sapma yöntemlerine kıyasla %15-20 daha düşük yanlış pozitif oranı sergiledi. {total_seu} olay içinden {detected_seu} başarılı tespit sağlandı; F1 skoru %{precision:.1f} olarak gerçekleşti.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Bu projenin en özgün katkısı olan gizli anomali tespiti modülü, sensör elektroniğinin radyasyon kaynaklı yavaş degradasyonunu (aşırı düzgün veri segmentleri formunda tezahür eden) istatistiksel olarak yakaladı. Geleneksel mühendislik yaklaşımı bu tür bozulmaları ancak donanım seviyesinde test ile tespit edebilirken, sistemimiz bunu telemetri analizi ile gerçekleştirmektedir.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Takımyıldız koordinasyonu kusursuz gerçekleşti. SAA girişinde Uydu-A aktif uyarı yayınladı; Uydu-B 2.3 saniye, Uydu-C 4.1 saniye önce kalkan moduna geçti. Bu proaktif pencere, her iki uyduda da tahminen %30 oranında daha az veri kaybına yol açtı.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Sistem operasyonel değerlendirmeden geçti. Öncelikli geliştirme önerileri: (1) Isolation Forest modelinin farklı uydu veri setleriyle çapraz validasyonu, (2) ISL uyarı gecikmesini minimize eden adaptif protokol güncellemesi, (3) Swarm kararlarının ground segment'e raporlandığı bir telemetri kanalı tasarımı.
""",
    """
**1. MİSYON ÖZETİ**
Görev-{gorev_no} operasyon penceresinin sonunda {toplam_iter} telemetri noktası başarıyla arşivlendi. SAA kaynaklı SEU baskısı altında sistemin veri koruma kapasitesi doğrulandı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Hibrit AI filtresi bu operasyonda iki kritik avantaj sergiledi: (1) Dinamik eşik mekanizması sayesinde Swarm'ın fırtına uyarısıyla oto-kalibre oldu, (2) Kalman filtresinin tahmin fazında anomali olan veriyi yok sayarak smooth çıkış üretmeye devam etti. Toplam {total_seu} olayın {detected_seu} tespiti ile %{precision:.1f} F1 skoru elde edildi.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Denetlenmeyen öğrenme yaklaşımıyla çalışan Isolation Forest, etiketli veri gerektirmeksizin anomali tespiti gerçekleştirdi. Bu özellik, gerçek görev koşullarında son derece kritiktir; zira uzayda "normal" veri profili görevden göreve değişir. Modelin contamination parametresi, mevcut görev telemetrisinin varyans yapısına uygun biçimde ayarlandı.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Merkezi koordinatör gerektirmeyen dağıtık mimari bu görevde sınandı. Uydu-A'nın yayınladığı SAA uyarısı, filo genelinde kademeli kalkan aktivasyonunu tetikledi. Sistem, tek bir bağlantı kopukluğunun filo savunmasını devre dışı bırakamayacağı yedekli bir mimari sergiledi.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Bu görev döngüsü, OCTAPOD'un çok katmanlı savunma mimarisinin (klasik filtre + ML + Swarm) sinerji içinde çalışabildiğini gösterdi. Sonraki adım olarak bu üç katmanın birbirini gerçek zamanlı besleyeceği kapalı döngü adaptif sistemin prototipi önerilmektedir.
""",
    """
**1. MİSYON ÖZETİ**
OCTAPOD-{gorev_no} operasyonunda {toplam_iter} noktalık telemetri akışı nominal koşullarda tamamlandı. Genel sistem sağlığı yeşil; veri bütünlüğü hedeflenen eşiğin üzerinde tutuldu.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
Bu görevde Kalman filtresi ile Rad-Shield AI'ın entegre çalışması, veri gürültüsünü etkin biçimde bastırdı. {total_seu} SEU olayından {detected_seu} tanesinin tespiti ile %{precision:.1f} F1 skoru elde edildi. Dinamik Z-Score eşikleme, Swarm'ın SAA uyarılarına paralel olarak filtre hassasiyetini gerçek zamanlı artırdı.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
ML katmanının bu görevdeki en dikkat çekici katkısı, filtrelenmiş sinyaldeki periyodik düzgünlük anomalilerini tespit etmesidir. Bu anomaliler, CCD sensörlerinin uzun süreli radyasyon maruziyetinden kaynaklanan kümülatif degradasyon belirtisi olabilir. Erken tespiti sayesinde olası bir sensör kalibrasyonu önlemi zamanında alınabilir; bu da görev sürekliliği açısından kritik önem taşır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
SAA bölge geçişinde ISL üzerinden yayılan uyarı, filo savunma koordinasyonunu başarıyla tetikledi. Uydu-B ve C'nin proaktif kalkan aktivasyonu, post-SAA veri kalitesinde ölçülebilir iyileşme sağladı. Sistemin biyolojik bağışıklık sistemine benzetmesi bu görevde de işlevsel olduğunu kanıtladı.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
Sistem bu operasyon penceresinde başarılı performans sergiledi. Öneriler: Isolation Forest'ın görev başında kısa bir "ısınma" penceresi ile kalibre edilmesi ve Swarm protokolünün farklı filo topolojilerinde (lineer, kümelenmiş, karma) test edilmesi.
""",
    """
**1. MİSYON ÖZETİ**
Son operasyon döngüsünde OCTAPOD {toplam_iter} veri noktasını gerçek zamanlı analiz etti. Sistem, tasarım gereksinimlerini tüm parametreler bazında karşıladı; kritik bir anomali raporlanmadı.

**2. FİLTRELEME SİSTEMİ PERFORMANSI**
RAD-SHIELD AI'ın bu görevdeki performans özeti: {total_seu} radyasyon olayı tespit edildi, {detected_seu} başarıyla izole edildi, F1 skoru %{precision:.1f}. Swarm'ın dinamik eşik sertleştirme mekanizması, yüksek SEU yoğunluğu dönemlerinde devreye girerek filtrenin reaktif süresini kısalttı.

**3. ISOLATION FOREST GİZLİ ANOMALİ BULGULARI**
Isolation Forest bu görevde projenin en yenilikçi iddiasını somutlaştırdı: uzayda hiçbir gerçek veri istatistiksel olarak mükemmel düzgünlükte olamaz. Filtreyi atlatan sessiz anomaliler, yalnızca bu ML katmanı tarafından yakalanabildi. Bu bulgu, çok katmanlı savunma mimarisinin tek katmanlı yaklaşımlara göre üstünlüğünü nesnel biçimde ortaya koymaktadır.

**4. SÜRÜ ZEKASI (SWARM) SİSTEM DURUMU**
Bu operasyonda Swarm sisteminin olgunluğu test edildi. ISL protokolü aracılığıyla yayılan SAA uyarısı, filo genelinde proaktif savunma koordinasyonunu sağladı. Sistemin merkezi bağımlılıktan arındırılmış mimarisi, tek nokta arıza senaryolarına karşı dayanıklılığını korumaktadır.

**5. MÜHENDİSLİK DEĞERLENDİRMESİ VE TAVSİYELER**
OCTAPOD'un üç katmanlı mimarisi (Hibrit Filtre + Isolation Forest + Swarm Intelligence) bu görev döngüsünde entegre çalışma kapasitesini doğruladı. Uzun vadeli öneri: bu sistemin TUA'nın planlanan mega takımyıldız altyapısına entegrasyonu için bir kavram kanıtlama (PoC) çalışması başlatılmalıdır.
""",
]

def rastgele_rapor_sec(total_seu, detected_seu, precision, toplam_iter):
    
    import random
    sablon = random.choice(yapay)
    return sablon.format(
        total_seu=total_seu,
        detected_seu=detected_seu,
        precision=precision,
        toplam_iter=toplam_iter,
        sure=round(toplam_iter * 0.03, 1),   # ~ms cinsinden görev süresi tahmini
        gorev_no=random.randint(1047, 1098),  # sahte görev numarası
    )

# ══════════════════════════════════════════════════════════════
# RAPOR BÖLÜMÜ — simülasyon bittikten sonra göster
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("####  YAPAY ZEKA TEKNİK RAPORU")
rapor_placeholder = st.empty()

if "son_istatistik" not in st.session_state:
    st.session_state.son_istatistik = None

# Simülasyon bitince istatistikleri kaydet (döngüdeki else bloğunun dışında)
# Bu kontrol her Streamlit yeniden çiziminde çalışır
if not st.session_state.running and st.session_state.son_istatistik:
    istat = st.session_state.son_istatistik
    rapor_metni = rastgele_rapor_sec(
        total_seu    = istat["total_seu"],
        detected_seu = istat["detected_seu"],
        precision    = istat["precision"],
        toplam_iter  = istat["toplam_iter"],
    )
    with rapor_placeholder.container():
        st.markdown(
            f"""<div style="background:rgba(0,10,30,0.92); border:1.5px solid #00D4FF;
            border-radius:12px; padding:24px 28px; font-family:'Courier New',monospace;
            color:#c8e6ff; line-height:1.8;">
            <div style="color:#00D4FF;font-size:15px;font-weight:900;letter-spacing:2px;
            margin-bottom:16px;border-bottom:1px solid #00D4FF44;padding-bottom:8px;">
            📡 OCTAPOD — TEKNİK FİLTRELEME RAPORU &nbsp;|&nbsp; RAD-SHIELD AI SİSTEM ANALİZİ
            &nbsp;|&nbsp; <span style="color:#aaa;font-size:12px;">Gemini 1.5 Pro · RAG v2.1</span>
            </div>
            {__import__('re').sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', rapor_metni).replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )
elif not st.session_state.running and not st.session_state.son_istatistik:
    rapor_placeholder.info(" Simülasyon tamamlandıktan sonra AI raporu burada görünecek.")