# Moonlight AI — Baholash

[README](../README.md) · [Loyiha xaritasi](PROJECT_GUIDE.md)

## Dalillar holati

Mavjud benchmark harness va suite ma’lumotlari: [BENCHMARKS.md](BENCHMARKS.md). Suite mavjudligi uning o‘tganini bildirmaydi. Ushbu hujjat yangi score e’lon qilmaydi.

## Nima o‘lchanadi?

| Yo‘nalish | Ushbu hujjatdagi holat |
|---|---|
| Held-out loss va suite score | O‘lchash va hisobotga yozish kerak |
| Tokens/s va peak RAM/VRAM | O‘lchash va hisobotga yozish kerak |
| Faol / umumiy parametrlar | O‘lchash va hisobotga yozish kerak |

## Qayta chiqarish protokoli

1. Commit SHA va branchni qayd eting.
2. OS, runtime, dependency versiyalari va apparatni yozing.
3. Ishlatilgan test ma’lumoti, konfiguratsiya va buyruqni saqlang.
4. Warm-up, takrorlash soni va o‘lchov chegarasini belgilang.
5. Taqqoslashda bir xil sharoitdan foydalaning; muhit farqlarini yozing.
6. Median, tarqoqlik, xatolar va o‘tkazib yuborilgan holatlarni ko‘rsating.

## Hisobot shakli

| Maydon | To‘ldiriladigan ma’lumot |
|---|---|
| Commit / konfiguratsiya | Aniq identifikator |
| Muhit | OS, runtime, qurilma, resurs limitlari |
| Ma’lumot / ssenariy | Manba, versiya, o‘lcham |
| Buyruq | Qayta bajariladigan aniq buyruq |
| Natija | Qiymat, birlik, takrorlar soni |
| Cheklov | O‘lchanmagan yoki qo‘llanmaydigan holatlar |

Test muvaffaqiyati, tezlik va foydalanuvchi sifati alohida xulosalardir. Lokal natijani production kafolati sifatida ko‘rsatmang. Maxfiy ma’lumotlarni hisobotga kiritmang.
