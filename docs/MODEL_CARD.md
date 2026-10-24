# Model Card

[README](../README.md) · [Baholash](EVALUATION.md)

## Identifikatsiya

Moonlight — model yaratish, trening va baholash kutubxonasi. Ushbu card aniq o‘qitilgan checkpointning yakuniy modeli emas. Konfiguratsiyada 1M–1T nominal variantlar bo‘lishi shu o‘lchamdagi model o‘qitilganini anglatmaydi.

## Arxitektura va maqsad

PyTorch asosidagi attention + retention model oilasi; dense va MoE variantlari konfiguratsiyaga bog‘liq. O‘zbek/ingliz matni, chat, kod va vosita ishlatish bo‘yicha tajribalar uchun. Tafsilot: [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGS.md](CONFIGS.md), [TRAINING.md](TRAINING.md).

## Checkpoint uchun to‘ldirilishi kerak

- Checkpoint nomi, SHA-256, source commit va to‘liq config.
- Haqiqiy total/active parametrlar va tokenizer versiyasi.
- Dataset manbalari, foydalanish huquqi, tozalash va train/eval ajratilishi.
- Tokenlar soni, trening bosqichlari, seed, apparat va hisoblash budjeti.
- [Benchmark](BENCHMARKS.md) natijalari, bajarilmagan vazifalar va cheklovlar.

## Natijalar va cheklovlar

Bu card yangi score yoki Transformer ustidan isbotlangan ustunlik da’vosini bermaydi. Random/untrained vaznlar tayyor assistant emas. Sintetik va offline test sifati real foydalanuvchi sifatidan ajratiladi. Chiqarilgan checkpoint uchun xavfsizlik, xatolik va til qamrovini alohida tekshiring.

## Litsenziya

Kutubxona litsenziyasi [LICENSE](../LICENSE) da. Dataset va tarqatiladigan vaznlar uchun huquqlarni alohida tasdiqlang.
