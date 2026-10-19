# Mercan Tool-Use Project Master Contract

**Dosya:** `PROJECT_MASTER.md`  
**Durum:** ACTIVE / AUTHORITATIVE  
**Sürüm:** 0.2  
**Tarih:** 2026-10-19  
**Ana kopya:** `SeperationOfPowers/PROJECT_MASTER.md` (git). TRUBA'daki `yasama-yurutme-yargi/PROJECT_MASTER.md` bunun kopyasıdır; değişiklik önce repoda yapılır. `MasterFormat.txt` tarihsel tasarım notudur; çelişkide bu dosya geçerlidir.

Bu dosya Mercan tool-use projesinin ortak çalışma sözleşmesidir. Yasama, Yürütme ve Yargı üzerinde yapılacak eğitim, veri hazırlama, evaluation ve runtime geliştirmelerinde bu dosya ortak referans olarak kullanılmalıdır.

Bu dosyanın amacı üç modülün bağımsız geliştirilirken birbirinden kopmasını önlemek, ortak formatı sabitlemek ve proje boyunca aynı kavramların farklı anlamlarda kullanılmasını engellemektir.

> Kural: Bu dosya ile çelişen yeni bir veri formatı, training target'ı, action mapping'i veya runtime protokolü doğrudan uygulanmamalıdır. Önce bu dosya güncellenmelidir.

---

# 1. Sistem Tanımı

Mercan tool-use sistemi üç mantıksal modülden oluşur:

1. **Yasama** — Decision Maker / Router
2. **Yürütme** — Argument Filler / Tool Caller
3. **Yargı** — Response Composer

Temel runtime akışı:

```text
USER
  ↓
YASAMA
  ↓ action
YÜRÜTME
  ↓ arguments
VALIDATOR / POLICY
  ↓
EXECUTOR
  ↓ observation
YASAMA
  ├─ yeni action → YÜRÜTME
  └─ FINISH → YARGI
               ↓
              USER
```

Sistem tek seferde statik plan üretmek zorunda değildir. Her executor observation'ından sonra Yasama tekrar karar verir. Mimari **reaktif agent döngüsü**dür.

---

# 2. Modüllerin Kesin Sorumlulukları

## 2.1 Yasama

Yasama'nın tek görevi:

```text
mevcut kullanıcı isteği + geçmiş/state + son observation
→ bir sonraki action
```

Yasama tool argümanı üretmez.

Doğru çıktı örneği:

```text
FILE_CREATE
```

Yanlış çıktı örneği:

```text
FILE_CREATE(path="notes.txt", content="Merhaba")
```

### Canonical 16-class ontology

```text
RESPOND
WEB_SEARCH
WEB_OPEN
FILE_READ
FILE_SEARCH
FILE_LIST
FILE_CREATE
FILE_EDIT
FILE_DELETE
FILE_MOVE
CODE_EXECUTE
SHELL_EXECUTE
DOWNLOAD
SEND_MAIL
ASK_CLARIFICATION
FINISH
```

Closed 16-class Yasama modeli yalnız açıkça bu ontology'ye uygun veya güvenli şekilde map edilmiş kayıtlarla eğitilmelidir.

Kaynak action semantic olarak birebir eşleşmiyorsa zorla 16 sınıftan birine map edilmemelidir.

Örnek güvenli alias'ler:

```text
web_search      → WEB_SEARCH
read_file       → FILE_READ
edit_file       → FILE_EDIT
list_directory  → FILE_LIST
search_code     → FILE_SEARCH
run_command     → SHELL_EXECUTE
```

Örnek zorlanmaması gerekenler:

```text
REQUEST_CONFIRMATION ≠ ASK_CLARIFICATION
send_message         ≠ SEND_MAIL
write_file           ≠ otomatik FILE_CREATE
calculator family    ≠ otomatik CODE_EXECUTE
```

Unmapped source action provenance için korunabilir; ancak canonical 16-class router eğitimine otomatik dahil edilmez.

---

## 2.2 Yürütme

Yürütme'nin görevi:

```text
user request
+ current state/context
+ Yasama'nın seçtiği action
+ tool schema / placeholder
→ JSON arguments
```

Yürütme doğal dil final cevap üretmez.

Örnek:

```text
ACTION: FILE_CREATE
PLACEHOLDER:
{"path": null, "content": null}
```

Target:

```json
{"path":"notes.txt","content":"Merhaba"}
```

Yürütme training target'ı gerçek JSON argument object'tir.

```python
target = json.dumps(row["target_arguments"], ensure_ascii=False)
```

Placeholder yaklaşımında model yeni key icat etmemeli, verilen schema/placeholder ile uyumlu alanları doldurmalıdır.

### Geçerli tool şemaları (v0.2)

Fiziksel registry şeması §14'te açık karar olarak duruyor; o karara kadar geçerli şemalar runtime registry'sidir (`runtime/local_tools.py` → `BUILTIN_TOOL_REGISTRY`). Dataset normalizasyonu bu alan adlarına göre yapılır. Zorunlu alanlar **kalın**.

| Action | Argümanlar | Not |
|---|---|---|
| FILE_LIST | `path`, `pattern`, `recursive` | `pattern` istekten çıkarılan glob'dur (`*.py`); filtre istenmediyse `*` |
| FILE_READ | **`path`**, `max_chars` | |
| FILE_SEARCH | **`query`**, `path`, `pattern`, `recursive`, `max_results` | Aranan metin `query`'dir; `pattern` dosya filtresidir |
| FILE_CREATE | **`path`**, **`content`** | |
| FILE_EDIT | **`path`**, **`old_text`**, **`new_text`** | |
| FILE_DELETE | **`path`** | |
| FILE_MOVE | **`source`**, **`destination`** | |
| WEB_SEARCH | **`query`**, `max_results` | Varsayılan ve üst sınır 3 (§6) |
| WEB_OPEN | **`url`** | Yalnız public http(s) |
| DOWNLOAD | **`url`**, `path` | |
| SHELL_EXECUTE | **`command`**, `cwd` | |
| CODE_EXECUTE | **`language`**, **`code`** | |
| SEND_MAIL | **`to`**, **`subject`**, **`body`** | `to` kullanıcının isteğinde açıkça geçmeli |

Kaynak verideki farklı alan adları bu tabloya normalize edilir; örneğin Gemini distil verisinde FILE_SEARCH'in aranan metni `pattern` alanında, DOWNLOAD'ın hedefi `output_path` / `destination` alanındadır.

Runtime'da Yürütme çıktısı doğrudan executor'a gönderilmez:

```text
JSON parser
→ schema validator
→ policy/safety validator
→ gerekiyorsa confirmation
→ executor
```

---

## 2.3 Yargı

Yargı'nın görevi:

```text
original/current user request
+ relevant conversation context
+ gerçek executor observation(lar)ı
→ final user-facing natural-language response
```

Yargı tool çağrısı üretmez ve executor'ın yapmadığı bir işlemi yapılmış gibi anlatmaz.

Temel kural:

> **Observation'da olmayan hiçbir başarıyı, sonucu veya bilgiyi uydurma.**

### Yargı target sözleşmesi

Disk üzerindeki dataset satırında alan adı `response` olabilir:

```json
{
  "user_prompt":"...",
  "observations":[...],
  "response":"Dosya başarıyla oluşturuldu."
}
```

Ancak training target wrapper değildir.

Doğru:

```python
target = row["response"]
```

Yanlış:

```python
target = json.dumps({"response": row["response"]})
```

Assistant span başladıktan sonra üretilen tokenlar doğrudan kullanıcıya gösterilecek final cevaptır.

---

# 3. Ortak Runtime Veri Sözleşmeleri

Üç modül aynı state ve observation semantiğini paylaşmalıdır.

## 3.1 Observation contract

Canonical observation şekli:

```json
{
  "action": "FILE_CREATE",
  "data": {},
  "status": "success",
  "error": null
}
```

Alanların anlamı:

```text
action → hangi action/tool çalıştı
data   → executor tarafından dönen gerçek payload
status → success | error
error  → hata yoksa null, varsa hata bilgisi
```

Bazı kaynaklarda ek provenance alanları bulunabilir:

```text
tool_call_id
source
provider
```

Bunlar canonical dört alanın semantiğini değiştirmez.

### Action'a göre `data` şekli (v0.2)

```text
FILE_LIST    → {path, pattern, recursive, entries: [...]}
FILE_READ    → {path, content, truncated}
FILE_SEARCH  → {query, matches: [{path, line, text}]}
WEB_SEARCH   → {query, results: [{title, url, snippet}]}     (en fazla 3 sonuç)
WEB_OPEN     → {url, title, text, truncated}
```

Hata durumunda `data` boş obje, `status: "error"`, `error` insan okunur hata metnidir. Executor'ı yapılandırılmamış bir action seçilirse runtime çökmez; `"<ACTION> aracı bu kurulumda yapılandırılmamış; çalıştırılamadı."` hata observation'ı üretir ve Yargı bunu kullanıcıya anlatır.

Kaynak veride farklı adlar normalize edilir; örneğin Gemini distil verisinde FILE_LIST listesi `files` alanındadır → `entries`.

## 3.2 State contract

Minimum ortak state:

```json
{
  "completed_actions": [],
  "last_observation": null
}
```

Gerektiğinde eklenebilir:

```text
recent_observations
step_index
turn_index
```

Ancak üç modül arasında aynı alan farklı anlamlarda kullanılmamalıdır.

Alan tipleri (v0.2):

```text
completed_actions   → bu turda çalışmış action adları, sırayla: ["FILE_LIST", "FILE_READ"]
last_observation    → son canonical observation veya null
recent_observations → bu turdaki observation'lar, sırayla
step_index          → bu turdaki karar adımı (0'dan başlar)
turn_index          → konuşma turu
```

`completed_actions` action **adı** listesidir. Gemini distil verisindeki `[{"action": ..., "status": ...}]` biçimi normalizasyonda ada indirgenir; status bilgisi observation'larda zaten vardır.

---

# 4. Training Serialization İlkeleri

Projede role-specific training yapılırken prompt ve target birbirinden açık şekilde ayrılmalıdır.

## Yasama

```text
INPUT:
user_prompt + conversation_history + state

TARGET:
target_action
```

```python
target = row["target_action"]
```

## Yürütme

```text
INPUT:
user_prompt + conversation_history + state + action + tool_schema/placeholder

TARGET:
target_arguments
```

```python
target = json.dumps(row["target_arguments"], ensure_ascii=False)
```

## Yargı

```text
INPUT:
user_prompt + relevant conversation_history + observations

TARGET:
response
```

```python
target = row["response"]
```

Yargı için observation input'tur, target değildir.

Yasama ve Yürütme için de source metadata veya verifier bilgisi target'a sızdırılmamalıdır.

## Fiziksel serialization (v0.2)

Prompt'un fiziksel metni tek kaynaktan üretilir: `runtime/serialization.py` içindeki `serialize_yasama_input`, `serialize_yurutme_input`, `serialize_yargi_input`. **Training verisi de bu fonksiyonlarla üretilir;** elle yazılmış ya da ayrı bir builder'da kopyalanmış bir template kullanılmaz. Böylece eğitimde görülen metin runtime'da görülen metinle birebir aynı kalır.

Son kullanıcı mesajının içeriği:

```text
Yasama  : USER:\n<istek>
          [CONVERSATION_HISTORY:\n<JSON>]          (geçmiş varsa)
          STATE:\n<JSON>

Yürütme : Yasama blokları
          + ACTION:\n<action>
          + TOOL_SCHEMA:\n<JSON>                   (registry'deki public alanlar)
          + PLACEHOLDER:\n<JSON>

Yargı   : <istek>
          <tool_observations>\n<JSON liste>\n</tool_observations>
```

Yargı'nın konuşma geçmişi user mesajına JSON olarak gömülmez; son 6 mesaj gerçek `kullanici` / `asistan` turları olarak verilir. Geçmişteki kullanıcı mesajları da aynı biçimde (`<istek>` + boş `<tool_observations>[]`) serialize edilir. Gerekçe: JSON bloğu küçük checkpoint'lerde cevap olarak geri kopyalandı ve her turda büyüdü.

Yasama'ya geçmişin hangi biçimde verileceği §14'te açık karardır.

---

# 5. ChatML / Assistant Span Kuralı

Decoder tabanlı modellerde assistant span başladıktan sonra üretilen içerik gerçek model çıktısıdır.

Rol başlıkları Türkçedir: `sistem`, `kullanici`, `asistan`. Bunlar Mercan native chat template'inin (libmercan `format_chat`, CLI, EthosoftLib) varsayılanlarıdır; training verisi de aynı başlıklarla serialize edilir. `system` / `user` / `assistant` yalnız mesaj listesindeki mantıksal rol adlarıdır.

Örnek Yargı serialization:

```text
<|im_start|>sistem
Sen Yargı modelisin. Kullanıcının isteğini ve gerçek araç gözlemlerini
kullanarak nihai cevabı üret. Gözlemlerde bulunmayan bilgi uydurma.
<|im_end|>
<|im_start|>kullanici
... request ...
<tool_observations>
[... gerçek observations ...]
</tool_observations>
<|im_end|>
<|im_start|>asistan
Gerçek final doğal dil cevabı.
<|im_end|>
```

Tokenizer'ında `<|im_start|>` / `<|im_end|>` special token olmayan checkpoint'ler bunları düz metin olarak üretebilir; runtime çıktıyı ilk `<|im_end|>` metninde keser. Fine-tune için special token içeren bir taban (örneğin `final_sft_corrected`) seçilmelidir.

Prompt/input tokenları role-specific SFT'de loss'tan maskelenmelidir. Final assistant target tokenları train edilmelidir.

Chat template yalnız system/user/assistant destekliyorsa yeni bir `tool` role uydurulmamalıdır; observations açık delimiter ile input içine gömülebilir.

---

# 6. Web Search ve Calculator Politikası

## Web Search

Büyük `arac_kullanimi` temizleme hattında web-search observation'ları kaynak rank sırasına göre kompaktlaştırılmıştır.

Aktif politika:

```text
web_search result listesi → en alakalı ilk 3 source/result
```

Amaç context şişmesini azaltırken retrieval kanıtını korumaktır.

Bu compaction yalnız observation içeriğini etkiler; final `response` target'ı değiştirilmez.

Runtime aynı sınırı uygular: WEB_SEARCH en fazla 3 sonuç döndürür (`max_results` varsayılanı ve üst sınırı 3). Arama yalnız yapılandırılmış bir SearXNG instance'ı üzerinden yapılır (`pipeline.searxng_url` veya `ETHOSOFT_SEARXNG_URL`); yapılandırılmadıysa web action'ları kapalıdır.

## Calculator

Calculator family source function/action adları semantic olarak korunur.

Örnekler:

```text
calc
calculator
calculate_bmi
calculate_discount
calculate_median
...
```

Bu action'lar otomatik olarak `CODE_EXECUTE`'a map edilmez.

Calculator observation/result değerleri truncation ile değiştirilmemelidir.

Runtime'da şu an hesaplama tool'u yoktur ve 16-class ontology'de karşılığı bulunmaz. `arac_kullanimi` içindeki calculator ailesi (~136k kayıt) tool listesi yeniden düzenlenene kadar canonical router eğitimine alınmaz (§14).

---

# 7. Mevcut TRUBA Dataset Gerçeği

Çalışma kökü:

```text
/arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi
```

Bu kökte birden fazla dataset jenerasyonu bulunmaktadır.

### Legacy seed

```text
master/
yasama/
yurutme/
yargi/
```

`train.jsonl`, `validation.jsonl`, `test.jsonl` dosyaları küçük legacy ToolCall-TR seed görünümüdür.

### Ayrı ek kaynaklar

```text
yasama/distil_data.jsonl
yurutme/distil_data.jsonl
yargi/distil_data.jsonl
```

Gemini distillation görünümüdür.

```text
master/hunter_tr.jsonl
yasama/hunter_tr.jsonl
yurutme/hunter_tr.jsonl
yargi/hunter_tr.jsonl
```

Hunter TR görünümüdür.

### `combined_v2/`

Aşağıdaki kaynakların eski birleşik snapshot'ıdır:

```text
legacy ToolCall-TR
+ hermes_tr
+ mobile_fc_tr
+ atasoglu_tool10k
+ itu_tool_tr
```

`combined_v2/` günümüzdeki bütün verinin authoritative birleşimi değildir.

Gemini, Hunter ve büyük `arac_kullanimi` corpusunun tamamı bu snapshot'a dahil değildir.

### `combined_v2_gz/` ve `combined_v2_sharded/`

Bunlar semantik olarak yeni datasetler değildir.

```text
combined_v2
→ gzip packaging
→ sharded gzip packaging
```

Aynı snapshot'ın storage varyantlarıdır.

### Büyük `arac_kullanimi` çalışma alanı

Raw/clean çalışma alanı ayrı dizindedir:

```text
/arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/arac_kullanimi_v1/
```

Bu veri Yasama/Yürütme/Yargı için büyük web-search + calculator kaynağıdır.

### Kapsam analizi (2026-10-19)

Mevcut kaynakların canonical 16-class kapsamı. Sayılar `target_action` değerlerinden alınmıştır.

| Kaynak | Yasama satırı | Canonical olanlar | Not |
|---|---|---|---|
| `arac_kullanimi` (HF `toolcall.parquet`) | 1,412,575 | FINISH 678,666 · WEB_SEARCH 597,740 | 136k calculator `unmapped_preserved`; geçmişli kayıt %0,6 |
| `combined_v2` (5 kaynak) | 182,123 | FINISH 27,145 · ASK_CLARIFICATION 1,468 · RESPOND 26 | 153k kayıt 16,545 farklı kaynak tool adında; `add_and_execute_jupyter_code_cell` 97k |
| Gemini distil | 2,814 | 16 sınıfın hepsi; dosya işlemlerinin tek kaynağı | geçmişli kayıt %0 |
| Hunter TR | 30,621 | SHELL_EXECUTE 3,110 · FILE_LIST 2,465 · FINISH 1,247 · FILE_EDIT 840 · FILE_READ 622 | `write_file` 22,274 korunmuş |

Toplamda zayıf kalan sınıflar: **RESPOND** (~60 kayıt), **WEB_OPEN**, **DOWNLOAD**, **SEND_MAIL**, **CODE_EXECUTE** (Jupyter dışı), dosya işlemleri (Gemini dışında yok) ve **çok turlu** kayıtlar. Bunlar runtime serializer'larıyla üretilecek sentetik veriyle tamamlanmalıdır.

Kaynak bazında kalite notları:

- `BTX24` mobil: telefon eylemleri ontology dışıdır; 1,464 ASK_CLARIFICATION kaydı kullanılabilir.
- `Tuguberk` hermes: Yargı'da 777 geçmişli kayıt — çok turlu Yargı için en iyi mevcut kaynak.
- `atasoglu`: Yargı görünümündeki 9,636 kaydın **tamamında observation boş**; tool sonucu olmadan cevap öğretme riski. İncelenmeden kullanılmaz.
- `ituperceptron`: Yargı cevapları `<answer>…</answer>` ile sarılı; etiket temizlenmeden kullanılmaz. `YASAMA_16CLASS_MAPPING_V1` bu kaynağın Jupyter çağrılarını CODE_EXECUTE'a map eder; uygulanırsa ~107k CODE_EXECUTE oluşur ve dağılım bozulur — örnekleme ile dengelenmelidir.
- Gemini distil Yargı: 11 kayıtta hedef cevap observation'lar içinde `RESPOND` olarak bulunuyor (target leakage); çıkarılmalıdır.

---

# 8. Şu Anda Authoritative Olmayan Şeyler

Aşağıdakiler tek başlarına "tüm güncel training datası" kabul edilmemelidir:

```text
yasama/train.jsonl
yurutme/train.jsonl
yargi/train.jsonl
combined_v2/*
combined_v2_gz/*
combined_v2_sharded/*
```

Bunların her biri belirli bir tarihsel snapshot veya source görünümüdür.

Yeni bir eğitim pipeline'ı bu dosyalardan birini rastgele seçip "full train" kabul etmemelidir.

---

# 9. Single Source of Truth Hedefi

Projenin bir sonraki veri-mühendisliği hedefi, tüm aktif kaynaklardan türetilen tek versioned shared training root oluşturmaktır.

Önerilen yapı:

```text
shared_data/
├── manifest.json
├── contracts/
│   ├── action_ontology.json
│   ├── observation_schema.json
│   └── state_schema.json
├── yasama/
│   ├── train.*
│   ├── validation.*
│   └── test.*
├── yurutme/
│   ├── train.*
│   ├── validation.*
│   └── test.*
└── yargi/
    ├── train.*
    ├── validation.*
    └── test.*
```

**Bu root henüz oluşturulmuş değildir.** Oluşturulana ve verifier ile ACTIVE ilan edilene kadar mevcut dataset parçaları source/provenance olarak değerlendirilmelidir.

`shared_data/manifest.json` gelecekte en az şunları sabitlemelidir:

```text
schema_version
dataset_version
created_at
source listesi
source revision/hash
row counts
split counts
dedup policy
ontology version
observation schema version
state schema version
role-specific serialization version
verifier result
```

---

# 10. Ortak Eğitim / Runtime Uyumluluk Kuralları

Üç modül ayrı ayrı geliştirilebilir; ancak aşağıdaki contract'lar ortak kalmalıdır:

1. Yasama'nın ürettiği action, Yürütme'nin beklediği action ile aynı string/ontology semantiğine sahip olmalıdır.
2. Yürütme'nin argümanları kullanılan tool schema ile doğrulanabilir olmalıdır.
3. Executor observation'ı canonical observation contract'a normalize edilmelidir.
4. Yasama tool sonrası karar verirken aynı observation semantiğini görmelidir.
5. Yargı yalnız gerçek observation'ı kanıt olarak kullanmalıdır.
6. Bir modülün eğitim formatı değişirse diğer modüllerin interface uyumluluğu kontrol edilmelidir.
7. Source action adı ile canonical ontology action birbirine karıştırılmamalıdır.
8. Provenance alanları korunabilir fakat model target'ına gereksiz şekilde taşınmamalıdır.

---

# 11. Runtime Güvenlik Zinciri

Hiçbir model executor üzerinde doğrudan sınırsız yetkiye sahip değildir.

Beklenen zincir:

```text
YASAMA
→ selected action

YÜRÜTME
→ candidate arguments

JSON/schema validator
→ schema doğru mu?

Policy layer
→ işlem izinli mi?

Confirmation layer
→ gerekiyorsa kullanıcı onayı var mı?

Executor
→ gerçek işlem

Observation normalizer
→ canonical observation

YASAMA
→ continue / FINISH

YARGI
→ final response
```

Minimum runtime korumaları:

```text
max step limit
repeated action+args limit
error-loop detection
destructive-action confirmation
path sandboxing
shell/code sandboxing
tool-output prompt injection protection
observation grounding
```

### Runtime durumu (v0.2)

| Koruma | Durum |
|---|---|
| max step limit | Var (`pipeline.max_steps`) |
| repeated action+args limit | Var: aynı action aynı argümanlarla tekrar seçilirse tool yeniden çalıştırılmaz, tur mevcut observation'larla Yargı'ya gider |
| JSON/schema validator | Var (`runtime/tool_validation.py`) |
| path sandboxing | Var: yerleşik okuma tool'ları çalışma klasörü dışına çıkamaz |
| WEB_OPEN ağ sınırı | Var: yalnız public http(s); loopback / özel adresler reddedilir |
| Eksik executor | Çökme yerine hata observation'ı (§3.1) |
| policy layer | **Yok** |
| destructive-action confirmation | **Yok** — FILE_DELETE / FILE_EDIT / SEND_MAIL vb. yalnız harici executor ile çalışır; onay o executor'ın sorumluluğundadır |
| error-loop detection | **Yok** (yalnız birebir tekrar yakalanıyor) |
| shell/code sandboxing | **Yok** — yerleşik executor yok |
| tool-output prompt injection protection | **Yok** — WEB_OPEN sayfa metni Yargı'ya olduğu gibi gider; öncelikli açık iş |

## Runtime'a özel geçici destekler

Aşağıdakiler eğitimsiz genel checkpoint'leri çalıştırmak için runtime'a eklenmiş geçici desteklerdir. **Training verisine girmezler** ve role özel fine-tune edilmiş checkpoint'lerde kaldırılmalıdırlar:

- **Few-shot turları:** Yasama, Yürütme ve (geçmiş yokken) Yargı prompt'larına önceki `kullanici` / `asistan` turları olarak örnekler eklenir. System prompt içine yazılan örnekler küçük modellerde kopyalandığı için tur biçimi seçildi. Yasama + örnekler ~2.8k token tutar.
- **Uzun system prompt kuralları:** Eğitilmiş modelde kısaltılmalıdır; §5'teki kısa prompt hedeftir.

---

# 12. Evaluation Sözleşmesi

## Yasama

```text
action accuracy
macro-F1
hard-pair accuracy
next-action accuracy
FINISH accuracy
clarification accuracy
```

Özellikle:

```text
FILE_READ vs FILE_SEARCH
FILE_CREATE vs FILE_EDIT
WEB_SEARCH vs WEB_OPEN
RESPOND vs TOOL
FINISH vs CONTINUE
```

## Yürütme

```text
JSON validity
exact schema match
per-field accuracy
whole-call accuracy
wrong-value leakage
negation/correction accuracy
missing-argument behavior
```

## Yargı

```text
observation faithfulness
success/error correctness
hallucination rate
JSON-copy rate
natural-response quality
critical-field recall
```

---

# 13. Değişiklik Yönetimi

Bu projede interface veya veri formatı değiştiren bir karar alınırken şu sıra izlenmelidir:

```text
1. PROJECT_MASTER.md contract güncellenir
2. schema/ontology version artırılır (gerekiyorsa)
3. dataset builder güncellenir
4. verifier güncellenir
5. shared dataset yeniden üretilir
6. role-specific training pipeline test edilir
7. runtime integration test edilir
```

Bir değişiklik yalnız tek modülde sessizce uygulanmamalıdır.

Örnek: observation schema değişirse yalnız Yargı formatter'ı değil, Yasama state builder ve runtime normalizer da kontrol edilmelidir.

---

# 14. Şu Anda Çözülmemiş / Sonraki Kararlar

Aşağıdaki konular henüz bu dosyada kesinleştirilmemiştir ve adım adım karara bağlanacaktır:

```text
- shared_data/ fiziksel dosya formatı (JSONL.GZ / Parquet / ikisi)
- exact active source listesi
- source balancing / sampling oranları
- Yasama'nın fiziksel model/checkpoint standardı
- Yürütme ve Yargı tek checkpoint mi iki checkpoint mi olacak
- tokenizer/chat-template kesin sürümü
- max sequence length ve truncation politikaları
- ortak experiment/run directory standardı
- checkpoint naming convention
- evaluator script contract'ları
- runtime service/process topolojisi
- tool registry'nin canonical fiziksel şeması
- tool listesinin yeniden düzenlenmesi (hesaplama tool'u dahil; calculator verisi buna bağlı)
- observation kısaltma politikası (aşağıda)
- Yasama'ya konuşma geçmişinin hangi biçimde verileceği (aşağıda)
```

Bu konularda karar verilene kadar mevcut davranış "resmi standart" gibi varsayılmamalıdır.

### Açık karar: observation kısaltma

Mevcut runtime davranışı (geçici; 4k context taşmasını önlemek için eklendi):

```text
Yasama / Yürütme STATE : string > 60 karakter → ilk 60 + "… (+N karakter)"
                         liste > 3 öğe       → ilk 3 + "… (+N öğe)"
Yargı observations     : string > 1500 karakter, liste > 10 öğe, aynı işaretle
```

Ölçülen ihtiyaç: tek bir orta boy dosya okuması (main.py, ~11k karakter) kısaltmasız ~7.5k token STATE üretiyordu ve iki kez taşınıyordu (`last_observation` + `recent_observations`). Kısaltmayla en büyük Yasama prompt'u 4096 içinde ~3.8k token.

Kullanıcı tercihi: içerik kesilmemeli. Karar verilecek alternatifler: kısaltmayı kaldırıp daha büyük context; yalnız Yasama/Yürütme'de içerik alanlarını çıkarıp Yargı'ya tam vermek; ya da mevcut kısaltmayı sözleşmeye almak. Karar verilene kadar training verisi bu kısaltma işaretlerini içermez.

### Açık karar: Yasama geçmiş biçimi

Mevcut runtime: konuşma geçmişi Yasama'ya `CONVERSATION_HISTORY` JSON bloğu olarak, mevcut turun adım zinciri `STATE` (completed_actions / observations) içinde verilir. Çok adımlı turlarda adım zincirinin, tek adımlı turlarda isteğin nasıl taşınacağı netleştirilecek.

---

# 15. Projenin Temel Prensibi

```text
YASAMA
Ne yapılmalı?

YÜRÜTME
Bu action hangi argümanlarla uygulanmalı?

EXECUTOR
Gerçekte ne oldu?

YARGI
Gerçekte olan şey kullanıcıya nasıl anlatılmalı?
```

Bu sınırlar korunursa üç modül bağımsız geliştirilebilir fakat birlikte uyumlu çalışabilir.

---

# 16. Kısa Kontrol Listesi

Yeni bir çalışma başlamadan önce:

```text
[ ] Hangi rol üzerinde çalışıyorum: Yasama / Yürütme / Yargı?
[ ] Input contract doğru mu?
[ ] Target yalnız o role ait target mı?
[ ] Canonical action ile source action'ı karıştırıyor muyum?
[ ] Observation gerçek executor/source observation mı?
[ ] Training target'a provenance veya wrapper sızıyor mu?
[ ] Kullandığım dataset gerçekten aktif/güncel kaynak mı?
[ ] Değişiklik diğer iki modülün interface'ini bozuyor mu?
[ ] Verifier bu değişikliği kontrol ediyor mu?
```

Bu kontrollerden biri belirsizse eğitim veya runtime değişikliğini doğrudan uygulamak yerine önce ortak contract netleştirilmelidir.

---

# 17. Değişiklik Kaydı

## 0.2 — 2026-10-19

- Ana kopya SeperationOfPowers reposuna taşındı; TRUBA kopyası buradan güncellenir.
- §2.2: Geçerli 13 tool şeması tabloya alındı (runtime registry).
- §3.1: Action'a göre observation `data` şekilleri; eksik executor hata observation'ı.
- §3.2: State alan tipleri; `completed_actions` action adı listesidir.
- §4: Fiziksel serialization tek kaynaktan (`runtime/serialization.py`); training verisi aynı fonksiyonlarla üretilir. Yargı geçmişi gerçek sohbet turları olarak.
- §5: Rol başlıkları `sistem` / `kullanici` / `asistan`; `<|im_end|>` düz metin kesme notu.
- §6: Runtime WEB_SEARCH en fazla 3 sonuç (veriyle aynı); runtime'da hesaplama tool'u yok.
- §7: Kaynak bazında 16-class kapsam analizi ve kalite notları.
- §11: Runtime koruma durumu; runtime'a özel geçici destekler (few-shot turları) training verisine girmez.
- §14: Açık kararlar — tool listesi yeniden düzenleme, observation kısaltma, Yasama geçmiş biçimi.
- Not: 0.2'deki runtime değişikliklerinin bir kısmı (few-shot turları, Yargı geçmiş biçimi, STATE kısaltma, yeni şemalar) §13'teki sıraya uyulmadan önce koda uygulanmıştı; bu sürüm sözleşmeyi koda yetiştirir.
