import numpy as np
import os

def load_nasa_sample(limit=1000):
    """NASA verisini train klasöründen çeker."""
    train_klasoru = 'data/data/train'
    try:
        if not os.exists(train_klasoru): return np.linspace(350, 450, limit)
        
        ilk_dosya_adi = os.listdir(train_klasoru)[0]
        dosya_yolu = os.path.join(train_klasoru, ilk_dosya_adi)
        veri = np.load(dosya_yolu)
        # İlk kolonu (genelde telemetri) ve ilk 'limit' kadarını al
        return veri[:, 0][:limit]
    except Exception:
        # Dosya bulunamazsa düz bir çizgi döndür (test için)
        return np.linspace(400, 400, limit)

def inject_radiation(sinyal, bozulma_orani=0.03, sicrama_gucu=50):
    """
    Kanka arkadaşının yazdığı 'radyasyon_carptir' fonksiyonu.
    Sinyale SEU (sıçrama) ve anlık çökme (0) ekler.
    """
    bozuk_sinyal = sinyal.copy()
    veri_sayisi = len(sinyal)
    anomali_sayisi = int(veri_sayisi * bozulma_orani)
    
    vurulan_noktalar = np.random.choice(veri_sayisi, anomali_sayisi, replace=False)
    
    for i in vurulan_noktalar:
        # Arkadaşının mantığı: %50 ihtimalle zıpla, %50 ihtimalle sıfıra çök
        if np.random.rand() > 0.5:
            bozuk_sinyal[i] += sicrama_gucu * np.random.choice([-1, 1])
        else:
            bozuk_sinyal[i] = 0 # Sensör anlık çöktü
            
    return bozuk_sinyal, vurulan_noktalar