# 1-2. hafta: kaynaklar ve çalışma sırası

Haftada yaklaşık 10 saat varsayımıyla. Tüm bağlantılar 15 Ekim 2026'da açılıp içerikleri kontrol edildi. Saatler tahmin; kod çalıştırarak okursan biraz uzar, ama asıl öğrenme orada.

## Kısa özet

| Konu | Ana kaynak | Neden bu | Süre |
| --- | --- | --- | --- |
| Sinyal temelleri | PySDR (ücretsiz online kitap, Python kodlu) | I/Q'yu bir yazılımcıya anlatmak için yazılmış; her bölümde numpy kodu var | 1. hafta ~4 sa, 2. hafta ~2 sa |
| Makale | O'Shea, Roy, Clancy 2018 (arXiv 1712.04578) | Kullandığın veri setinin ve ResNet baseline'ının kaynağı | ~2 sa |
| PyTorch | Resmi "Learn the Basics" serisi + repodaki `train.py` | Kısa, güncel, tam olarak bizim kullandığımız API'ler | 1. hafta ~2 sa, 2. hafta ~1 sa |
| Overfitting / eğitim disiplini | Karpathy "A Recipe for Training Neural Networks" + d2l.ai 3.6 | Mülakatta "modelin neden iyi/kötü" sorusuna cevap verdiren iki metin | ~1.5 sa |
| Veri + GPU | DeepSig indirme sayfası, ASU Research Computing (Sol), Colab SSS | Kurulum kararları | ~1-2 sa (+ indirme süresi) |

---

## 1. hafta (12-18 Ekim): temel + kurulum

### A. Önce bugün başlat (5 dk iş, uzun bekleme)

1. **Sol hesabı iste.** ASU'nun süper bilgisayarı Sol öğrencilere ücretsiz ve A100/H100 GPU'ları var ([About](https://docs.rc.asu.edu/about)). Ama öğrencilerin **bir öğretim üyesi tarafından sponsor edilmesi gerekiyor** ([Getting Access](https://docs.rc.asu.edu/getting-access)). Bir hocaya kısa mail at ("kişisel bir ML projesi için Sol hesabıma sponsor olur musunuz?"), sonra [hesap formunu](https://docs.rc.asu.edu/voyager-accounts) doldur. Hesap genelde iki iş günü içinde açılıyor. Yeni kullanıcı rehberi: [New User Guide](https://docs.rc.asu.edu/new-user-guide).
2. **Veriyi indirmeye başla.** [DeepSig datasets](https://www.deepsig.ai/datasets/) sayfasında "RADIOML 2018.01A" altında form gerektirmeyen direkt link var: `https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz`. Lisans CC BY-NC-SA 4.0 (ticari olmayan kullanım serbest; README'de makaleye atıf ver). DeepSig sayfada setin "bilinen hataları (errata)" olduğunu açıkça yazıyor; sınıf sırası sorunu bunlardan biri, 2. haftada onu doğruluyoruz.

**GPU kararı için önerim: veriyi Sol'da (ya da kendi bilgisayarında) tut, Colab'a sadece alt kümeyi taşı.** Sebep: tam dosya ~21 GB; ücretsiz Google Drive 15 GB, ve Colab SSS'ye göre Drive alanı almak Colab'ın disk alanını büyütmüyor, ücretsiz oturumlar en fazla 12 saat sürüyor ve GPU erişimi "heavily restricted" ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). Buna karşılık `rfmodrec.prepare --per-class-snr 256` çıktısı yaklaşık 1.3 GB (24 sınıf × 26 SNR × 256 kare × 1024 × 2 × 4 bayt), 1024'lük alt küme yaklaşık 5.2 GB. Yani Sol açılana kadar Colab yeterli, 4. haftadan sonra Sol daha rahat.

### B. Sinyal temelleri: PySDR (~4 saat)

[PySDR](https://www.pysdr.org) Marc Lichtman'ın ücretsiz kitabı. SDR donanımı bölümlerini (Pluto, USRP, RTL-SDR...) atla, bize gerekmiyor.

| Sıra | Bölüm | Odaklan | Süre |
| --- | --- | --- | --- |
| 1 | [2. Frequency Domain](https://www.pysdr.org/content/frequency_domain.html) | FFT, negatif frekans neden var. Hızlı geç. | 45 dk |
| 2 | [3. IQ Sampling](https://www.pysdr.org/content/sampling.html) | Quadrature Sampling, Complex Numbers, Carrier and Downconversion, **Baseband and Bandpass Signals**, Calculating Average Power. "Receiver Architectures" ve "DC Spike" atlanabilir. | 1.5 sa |
| 3 | [4. Digital Modulation](https://www.pysdr.org/content/digital_modulation.html) | ASK, PSK, **IQ Plots/Constellations**, QAM, FSK. Sondaki "Python Example"ı mutlaka çalıştır: QPSK üretip AWGN ve faz gürültüsü ekliyor, 2. haftadaki konstelasyon işinin aynısı. | 1.5 sa |
| 4 | [10. Noise and Random Variables](https://www.pysdr.org/content/noise.html) | Decibels, Complex Noise, AWGN, **SNR and SINR**. "Deeper Dive into Random Variables" kısmını atla. | 45 dk |

Bu bölümden sonra şunları kendi cümlelerinle söyleyebilmelisin: bir I/Q örneği neden karmaşık sayı; 16QAM'de neden 16 nokta var ve 64QAM'den neden düşük SNR'da zor ayrılıyor; 0 dB SNR ne demek.

### C. Makale: O'Shea vd. 2018 (~2 saat)

[arXiv 1712.04578](https://arxiv.org/abs/1712.04578), 13 sayfa. Hepsini aynı dikkatle okuma:

- **Oku:** II.B Radio Channel Models (veriye eklenen bozulmalar: frekans kayması, sembol hızı kayması, çok yollu sönümleme, AWGN), III Dataset Generation (24 sınıflık "zor" set), IV.B-C CNN ve ResNet (Tablo III ve IV, bizim `models.py`'nin karşılığı), V.E sınıf bazında performans ve karışıklık matrisleri, V.F eğitim verisi miktarı.
- **Göz gezdir:** II.A baseline (istatistiksel öznitelik + gradient boosting), V.G-H havadan ölçüm ve transfer learning.
- **Akılda tut:** 24 sınıfta yüksek SNR'da yaklaşık %95 doğruluk raporluyorlar (V.F). Bizim hedef tablomuz bununla kıyaslanacak.

### D. PyTorch temeli (~2 saat)

[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) serisinden dördü yeter:

1. [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) (20 dk)
2. [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) (30 dk). Okuduktan sonra `src/rfmodrec/data.py` içindeki Dataset sınıfına bak; aynı `__len__` / `__getitem__` kalıbı.
3. [Build Model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) (20 dk). Sonra `models.py`'deki CNN'e bak. Girdi şekli `(B, 2, 1024)`: 2 kanal = I ve Q.
4. [Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) (30 dk). Sonra `train.py`'yi baştan sona oku; train/val döngüsü, early stopping (`--patience`), en iyi modeli kaydetme.

Video tercih edersen aynı konuların resmi video serisi: [Introduction to PyTorch - YouTube Series](https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html), özellikle 6. bölüm "Training with PyTorch".

### E. 1. haftanın bitiş kontrolü (PLAN.md'deki gibi)

- `pytest -q` geçiyor.
- `PYTHONPATH=src python -m rfmodrec.train --synthetic --model cnn --epochs 10` çalıştı ve `runs/` altında grafikler çıktı.
- Veri diskte (Sol, kendi bilgisayarın ya da harici disk).

---

## 2. hafta (19-25 Ekim): sınıf sırası kontrolü

### A. Biraz daha sinyal (~2 saat)

| Bölüm | Neden | Süre |
| --- | --- | --- |
| [17. Synchronization](https://www.pysdr.org/content/sync.html), sadece "Simulating Wireless Channel" ve "Adding a Frequency Offset" | RadioML'de frekans kayması var; kayma konstelasyonu **döndürür**. Yüksek SNR'da bile noktalar halka gibi görünebilir, şaşırma. | 45 dk |
| [16. Pulse Shaping](https://www.pysdr.org/content/pulse_shaping.html), göz gezdir | Neden gerçek sinyalde noktalar arasında "geçiş" örnekleri görüyorsun. | 30 dk |
| [4. Digital Modulation](https://www.pysdr.org/content/digital_modulation.html), FSK kısmındaki AM/FM animasyonu | Analog sınıflar (AM-SSB/DSB, FM) için. PySDR bunlara ayrı bölüm ayırmıyor; aşağıdaki pratik ipuçları bu açığı kapatıyor. | 15 dk |

### B. Eğitim disiplini (~1.5 saat)

- [Karpathy, "A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) (45 dk). Özellikle "Become one with the data" (2. haftanın işi tam olarak bu) ve "Overfit" kısmı.
- [d2l.ai 3.6 Generalization](https://d2l.ai/chapter_linear-regression/generalization.html) (45 dk). Training ve generalization error, underfitting/overfitting, validation set neden var.
- Küçük deney: sentetik veride 200 örneklik bir alt kümeyle eğit ve train loss'un sıfıra indiğini gör. Bu "model ve döngü doğru kurulmuş mu" testidir; Karpathy'nin önerdiği ilk adım.

### C. Asıl iş: konstelasyonlarla sınıf sırasını doğrula (~4-5 saat)

1. `PYTHONPATH=src python -m rfmodrec.prepare <hdf5> data/radioml_subset --per-class-snr 256`
2. Her etiket için SNR ≥ 20 dB olan birkaç kareyi seç ve I'ya karşı Q'yu scatter olarak çiz (24 küçük grafik, 4×6 ızgara).
3. Ne görmen gerektiği:
   - **OOK, 4ASK, 8ASK:** noktalar tek bir doğru üzerinde (OOK'ta biri sıfırda). Kayma varsa doğru döner ama yine doğrudur.
   - **BPSK/QPSK/8PSK/16PSK/32PSK:** tek bir çember üstünde 2/4/8/16/32 nokta. Genlik sabit.
   - **APSK:** iç içe çemberler (halkalar).
   - **QAM:** kare ızgara. 16 / 32 / 64 / 128 / 256 nokta sayısıyla ayrılır; yüksek dereceler yoğun bulut gibi görünür.
   - **GMSK, OQPSK, FM:** konstelasyonları "nokta" değil, çember üstünde sürekli bir iz.
   - **AM sınıfları:** konstelasyon pek bilgi vermez; zaman ekseninde `np.abs(x)` (zarf) çiz. DSB-WC'de taşıyıcı olduğu için zarf sıfıra inmez.
4. Döndürme kafanı karıştırırsa genliğin histogramına bak (`np.abs(x)`): PSK tek tepe, APSK birkaç tepe, QAM daha çok tepe verir. Kayma genliği değiştirmez.
5. Sonucu `data.RADIOML_2018_CLASSES` ile karşılaştır, uyuşmayan varsa not al. Bitiş: konstelasyon figürü + doğrulanmış sınıf listesi.

Takıldığın anda figürü bu thread'e at, birlikte bakarız.

---

## İsteğe bağlı ek okuma

- O'Shea, Corgan, Clancy 2016, "Convolutional Radio Modulation Recognition Networks" ([arXiv 1602.04105](https://arxiv.org/abs/1602.04105)): aynı grubun ilk CNN makalesi, 11 sınıflık RadioML 2016 seti. Mülakatta "bu alandaki ilk çalışma" olarak anılır.
- Sol kullanmaya başlayınca: [Sol web portalı](https://sol.asu.edu) (tarayıcıdan Jupyter) ve [ofis saatleri / yardım](https://docs.rc.asu.edu/contact-us).
