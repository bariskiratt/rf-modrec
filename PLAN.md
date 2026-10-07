# ML projesi iş listesi (12 Ekim 2026 – 31 Ocak 2027)

Haftada yaklaşık 7 saat proje varsayımıyla 16 takvim haftası, finaller çıkınca yaklaşık 13 haftalık aktif iş. Hedef: Şubat başında herkese açık GitHub reposu, sonuç tablosu dolu README ve 4 sayfalık İngilizce teknik rapor.

| Hafta | Başlangıç | İş | Bitti sayılması için |
| --- | --- | --- | --- |
| 1 | 12 Eki | O'Shea vd. 2018 makalesini oku. Repoyu kendi GitHub'ına koy, `pytest` ve sentetik eğitimi çalıştır. DeepSig'den RadioML 2018.01A'yı indir (~21 GB). GPU kaynağını seç (Colab veya ASU Research Computing). | Testler geçiyor, veri diskte |
| 2 | 19 Eki | `rfmodrec.prepare` ile 256'lık alt küme çıkar. Her etiket için yüksek SNR konstelasyonlarını çiz ve sınıf sırasını doğrula. | Konstelasyon figürü, doğrulanmış sınıf listesi |
| 3 | 26 Eki | CNN baseline'ı alt kümede eğit. İlk doğruluk–SNR eğrisi. | `runs/cnn-*` ve eğri |
| 4 | 2 Kas | 1024'lük alt kümeye geç, öğrenme oranı ve batch boyutunu ayarla. Baseline'ı README'ye yaz. | README'de ilk satır |
| 5 | 9 Kas | ResNet'i aynı bütçeyle eğit, CNN ile karşılaştır. | Tabloda iki satır |
| 6 | 16 Kas | Transformer'ı eğit; patch boyutu 8 / 16 / 32 karşılaştırması. | Tabloda üç satır + kısa ablation |
| 7 | 23 Kas | Düşük SNR analizi: karışıklık matrisleri, sınıf bazında doğruluk, en çok karışan sınıf çiftleri. | Analiz figürleri |
| 8 | 30 Kas | Dayanıklılık deneyi: faz döndürme ve frekans kayması ile veri artırımı. | Artırımlı / artırımsız karşılaştırma |
| 9 | 7 Ara | Finaller. Sadece o ana kadarki sonuçları README'ye işle. | |
| 10 | 14 Ara | Finaller. Ara. | |
| 11 | 21 Ara | En iyi modeli int8'e quantize et, ONNX'e aktar, CPU'da ms/kare ölç. | Tabloda "int8" satırı |
| 12 | 28 Ara | Tampon hafta; isteğe bağlı: az örnekle (few-shot) yeni sınıf ekleme denemesi. | |
| 13 | 4 Oca | README'yi bitir: sonuç tablosu, figürler, "neden bu problem" bölümü. | README tamam |
| 14 | 11 Oca | 4 sayfalık İngilizce teknik rapor (giriş, yöntem, sonuçlar, sınırlamalar). | PDF repoda |
| 15 | 18 Oca | Kısa demo (notebook veya 2 dakikalık video), kod temizliği. | Demo linki |
| 16 | 25 Oca | Repoyu herkese açık yap, CV'ye ekle, 2 dakikalık proje anlatımını prova et. | CV'de proje satırı |

## Mülakatta anlatılacak üç cümle

1. Ne yaptın: ham I/Q sinyalinden modülasyon türünü tanıyan üç model ailesini karşılaştırdım.
2. Ne buldun: hafta 7–8 sonuçlarından gelecek (ör. hangi SNR'da hangi model kırılıyor).
3. Neden ASELSAN için önemli: elektronik destek ve spektrum izlemede bilinmeyen yayıncıyı tanımanın ilk adımı; int8 sonucu gömülü sistemde çalışabilirliği gösteriyor.
