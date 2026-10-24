# Mercan Tool-Use Project Master Contract

**Dosya:** `PROJECT_MASTER.md`  
**Durum:** ACTIVE / AUTHORITATIVE  
**Sürüm:** 0.11  
**Tarih:** 2026-10-24  
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

### Canonical 17-class ontology

```text
RESPOND
WEB_SEARCH
WEB_OPEN
FILE_READ
FILE_SEARCH
FILE_LIST
CALCULATE
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

Closed 17-class Yasama modeli yalnız açıkça bu ontology'ye uygun veya güvenli şekilde map edilmiş kayıtlarla eğitilmelidir.

Kaynak action semantic olarak birebir eşleşmiyorsa zorla 17 sınıftan birine map edilmemelidir.

Örnek güvenli alias'ler:

```text
web_search      → WEB_SEARCH
read_file       → FILE_READ
edit_file       → FILE_EDIT
list_directory  → FILE_LIST
search_code     → FILE_SEARCH
run_command     → SHELL_EXECUTE
calc            → CALCULATE      (expression aynen)
calculator      → CALCULATE      (operation/a/b → expression — §6)
write_file      → FILE_CREATE    (koşullu, yalnız yeni dosya yazan adımlar — aşağıda)
```

Örnek zorlanmaması gerekenler:

```text
REQUEST_CONFIRMATION ≠ ASK_CLARIFICATION
send_message         ≠ SEND_MAIL
write_file           ≠ otomatik FILE_CREATE
calculator family    ≠ otomatik CODE_EXECUTE
calculate_bmi, calculate_loan_payment, ... ≠ otomatik CALCULATE  (alan formülleri; §6)
```

**Koşullu alias — `write_file` → FILE_CREATE (Hunter TR, 0.7):** Araç mesajı yeni dosya / üzerine yazma ayrımı yapmaz ("Successfully wrote N characters to X"), ama her Hunter oturumu boş bir klasörde başlar (listelerde yalnız `session_state.json`). Bir `write_file` adımı FILE_CREATE sayılır: yolu o konuşmada daha önce yazılmamış, okunmamış veya listelenmemişse **ve** öncesinde hiç `run_command` çalışmamışsa (`npm create vite` gibi iskele komutları dosya yaratır, ajan sonra üzerine yazar). Ölçüm: 22,259 çağrının 11,066'sı FILE_CREATE; 11,013'ü bir komuttan sonra (belirsiz), 180'i kesin üzerine yazma. Diğerleri Yasama'da ontology dışı kalır; dosyanın tamamını değiştirdikleri için `old_text/new_text` FILE_EDIT'ine karşılık gelmezler. Eşleme adıma bağlıdır: hedefte, adım zincirinde ve observation'larda aynı adım için uygulanır (`datatools/normalize.py` → `new_file_write_steps`).

Unmapped source action provenance için korunabilir; ancak canonical 17-class router eğitimine otomatik dahil edilmez.

CALCULATE 0.4'te 17. sınıf olarak eklendi (§6). §7'deki kapsam sayıları eklemeden önceki 16 sınıfa göredir.

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

### Geçerli tool şemaları (v0.5)

Tek kaynak koddur: `runtime/local_tools.py` → `BUILTIN_TOOL_REGISTRY` ve Yasama'nın action sırası (`yasama/yasama.py` → `ACTION_ORDER`, `CONTROL_ACTION_DESCRIPTIONS`). Fiziksel registry bu koddan üretilen **`runtime/tool_registry.json`** dosyasıdır; veri hatları ve belgeler bu dosyayı okur, elle düzenlenmez (`python -m runtime.registry_export`; `--check` eskimişse 1 döner, testler de kontrol eder).

```text
{
  "format": 1,
  "contract": "<bu dosyanın sürümü>",
  "actions": [17 action, Yasama prompt sırası],
  "control_actions": {RESPOND | ASK_CLARIFICATION | FINISH: açıklama},
  "tools": {ACTION: {description, parameters (JSON Schema object), placeholder,
                     executor: "local" | "web" | "external"}}
}
```

`executor`: `local` her kurulumda süreç içinde çalışır; `web` SearXNG adresi verilince yerleşiktir; `external` için `--executor` gerekir. Dataset normalizasyonu bu alan adlarına göre yapılır. Zorunlu alanlar **kalın**.

| Action | Argümanlar | Not |
|---|---|---|
| FILE_LIST | `path`, `pattern`, `recursive` | `pattern` istekten çıkarılan glob'dur (`*.py`); filtre istenmediyse `*` |
| FILE_READ | **`path`**, `query` | Uzun dosya/PDF'te aranacak konu; boşsa kullanıcının isteği (§3.1) |
| FILE_SEARCH | **`query`**, `path`, `pattern`, `recursive`, `max_results` | Aranan metin `query`'dir; `pattern` dosya filtresidir |
| CALCULATE | **`expression`** | Rakam, `+ - * / ^`, parantez ve `sqrt()`; yazıyla yazılmış sayı rakama çevrilir (§6) |
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

Kaynak verideki farklı alan adları bu tabloya normalize edilir (`datatools/normalize.py` → `normalize_arguments`; ölçüm §7):

| Action | Kaynak alanı → registry | Atılan | Kaynak |
|---|---|---|---|
| FILE_LIST | `dir_path`, `directory` → `path` | | Hunter |
| FILE_READ | `file_path` → `path` | `limit`, `offset` (uzun dosya RAG ile okunur) | Hunter |
| FILE_EDIT | `file_path` → `path` | | Hunter |
| FILE_SEARCH | `pattern` (`query` yoksa) → `query`; `file_pattern` → `pattern`; `extension` → `pattern: "*.ext"` | | Gemini |
| FILE_MOVE | `source_path` → `source`, `destination_path` → `destination` | | Gemini |
| FILE_DELETE | tek elemanlı `paths` → `path` | | Gemini |
| SEND_MAIL | `recipient` → `to`, `message` → `body` | | combined_v2 |
| DOWNLOAD | `output_path`, `destination`, `target_directory` → `path` | | combined_v2, Gemini |
| WEB_SEARCH | `q` → `query`; `num`, `limit` → `max_results` (≤ 3) | `related_keywords`, `country`, `lang`, `gl`, `lr`, `start` | combined_v2 |
| SHELL_EXECUTE | | `timeout` | Hunter |
| CODE_EXECUTE | `language` yoksa `"python"` | `answer` (yalnız `null` ise; doluysa kayıt atılır) | ituperceptron |
| CALCULATE | `{operation, a, b}` → `expression` (§6) | | arac_kullanimi |

Dönüşümden sonra argümanlar registry şemasıyla doğrulanır; geçmeyen kayıt sebebiyle sayılır, zorlanmaz. Eğitim verisinde FILE_CREATE `content` en fazla 7000 karakterdir (8k bağlam).

**Açık araçlar (Yürütme ve Yargı, 0.7):** Hedef kullanım şirkete özel araçlardır: Yasama şirketin araç kümesine göre adapte edilir, Yürütme kendisine verilen *her* şemayı doldurur, Yargı dönen *her* sonucu özetler. Bu yüzden registry dışı kaynak araçları Yürütme ve Yargı verisinden atılmaz:

- Yürütme: araç kaynak adıyla `ACTION` olur; kaynak şeması runtime'ın registry araçlarına verdiği biçime çevrilir: `{description, parameters, placeholder}`. Kabul edilen kaynak biçimleri OpenAI `{"type": "function", "function": {...}}` ve `{name, description, parameters}` (yanlış yerdeki üst düzey `required` `parameters`'a taşınır). Argümanlar o şemaya göre doğrulanır. Şeması olmayan kaynaklar (ituperceptron'un Jupyter dışı araçları, BTX24 mobil) şemayı argümandan çıkarmak hedefi sızdıracağı için alınmaz.
- Yargı: observation kaynak araç adını korur; yalnız zarf normalize edilir (`tool_call_id` atılır, Hunter kabuğu açılır).
- Yasama: 17 sınıflık ontology'de kalır.
- Her rolde mesajları toplam 16,000 karakteri (~6–7.7k token) aşan kayıt atılır.

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

### Action'a göre `data` şekli (v0.4)

```text
FILE_LIST    → {path, pattern, recursive, entries: [...]}
FILE_READ    → {path, content}                                   (dosya ≤ 7000 karakter)
FILE_READ    → {path, content, partial: true, query, matched, total_chars}   (uzun dosya / PDF)
CALCULATE    → {result}                                          (number)
FILE_SEARCH  → {query, matches: [{path, line, text}]}
WEB_SEARCH   → {query, results: [{title?, url, snippet}]}    (en fazla 3 sonuç; `title` runtime'da var, veride yok)
WEB_OPEN     → {url, title, text, truncated}
```

Uzun FILE_READ (RAG): dosya `ethosoftlib.rag.Index` (BM25, Türkçe 5 harf kök) ile ~120 kelimelik parçalara bölünür; sorguyla en ilgili parçalar toplam 7000 karakteri geçmeyecek kadar seçilir ve dosyadaki sırayla `content` içinde verilir. Her parçanın başında `[satır N]` (PDF'te `[sayfa P, satır N]`) bulunur. Sorgu, Yürütme'nin `query`'si + kullanıcının isteğidir; okunan dosyanın yazılı adı ve istek kelimeleri (oku, dosya, ne diyor...) çıkarılır. Hiçbir terim eşleşmezse dosyanın başı verilir ve `matched: false` olur. Bölümler ayrı obje listesi değil tek `content` metnidir: training verisindeki FILE_READ observation'ları `{path, content}` biçimindedir ve liste verildiğinde Yargı onu JSON olarak geri döktü. 7000 karakter ölçümle seçildi: serialize edilmiş observation 2.1–2.7 karakter/token, yani ≤ ~3.4k token; Yargı (~2.4k prompt + 768 cevap) ve Yürütme (~2.9k + 512) 8192'ye ikinci küçük bir observation ile sığar.

CALCULATE ifadeyi `ethosoftlib.math.evaluate` ile hesaplar: kendi ayrıştırıcısıdır, Python `eval` kullanmaz (`import os`, `__import__` reddedilir). Kesin sonuç (kesir) sayıya çevrilir: tam sayılar 10^21'e kadar tam, diğerleri 12 anlamlı basamak (0.1 + 0.2 → 0.3, 340/80 → 4.25); 10^21 ve üstü bilimsel gösterim. Sıfıra bölme, çok büyük üs, bilinmeyen içeren ifade (`x+1`) ve anlaşılamayan ifade hata observation'ıdır. `ethosoftlib[math]` (sympy) kurulu değilse CALCULATE hata observation'ı döner; pipeline'ın kendisi bağımlılıksızdır.

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

### Rollerin gördüğü observation ayrıntısı (v0.3)

Observation'lar **kesilmez**. Her rol işi için gereken ayrıntıyı görür:

```text
Yasama  : STATE içindeki her observation {action, status, error}; son üç observation
          ayrıca data_preview = data'nın kompakt JSON'unun ilk 200 karakteri ("…" ile)
          (0.11; tam içerik yok — devam / bitir kararı için sonucun başı yeter)
Yürütme : STATE içindeki observation'lar tam (önceki sonuçtan argüman çıkarır,
          örn. FILE_READ sonrası FILE_EDIT'in old_text'i)
Yargı   : <tool_observations> tam
```

Serialize edilen STATE'te `last_observation`, `recent_observations` listesinin son öğesiyle aynıysa yazılmaz; aynı içerik bir kez taşınır. Farklıysa korunur. Pipeline'ın kendi state nesnesi değişmez; bu yalnız modele giden metnin biçimidir.

Uygulama: `runtime/serialization.py` → `yasama_state`, `yurutme_state`.

Context: libmercan rolleri varsayılan **8192** token context ile açılır (config'te `n_ctx` verilirse o geçerlidir). Mercan/NedoLM 0.8B checkpoint'leri GGUF metadata'sında 32768 context bildirir. 8192'ye sığmayan içerik kesilmez: uzun dosyalar FILE_READ'in RAG modunda soruyla ilgili bölümleriyle okunur (§3.1); `offset` ile parça parça okuma yerine bu seçildi.

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

Yasama'nın konuşma geçmişi (v0.3): `CONVERSATION_HISTORY` bloğu yalnız turun **ilk adımında** (henüz tool çalışmamışken, `completed_actions` boş) verilir; "aynısını tests için de yap" gibi istekler bu adımda çözülür. Bir tool çalıştıktan sonraki adımlarda Yasama yalnız USER + adım zinciri (STATE) görür; karar artık "mevcut istek bitti mi" sorusudur ve önceki turlar yalnız eski action'ların kopyalanmasına yol açar. Yürütme her adımda geçmişi görmeye devam eder.


## Eğitim system prompt'u (0.9)

Her rolün eğitim ve runtime'da **aynı** system prompt'u vardır: `yasama.TRAINED_SYSTEM_PROMPT`, `yurutme.TRAINED_SYSTEM_PROMPT`, `yargi.TRAINED_SYSTEM_PROMPT` (`system_message()`). Eğitim karışımı (§9.1) her kaydın başına bunu koyar; runtime bunu `roles.<rol>.prompt_style: "trained"` ile gönderir ve few-shot turlarını eklemez. Varsayılan `"few_shot"`, eğitilmemiş genel checkpoint'ler içindir (eski uzun prompt + örnek turlar). Prompt değişirse mix yeniden üretilir ve model yeniden eğitilir; manifest prompt'un sha256'sını tutar.

- Yasama (1,200 karakter, ~384 token; 0.10): görev ve çıktı biçimi, 17 action'ın her biri için tek satırlık "ne zaman seçilir" özeti (`yasama.ACTION_SUMMARIES`) ve üç karar kuralı (USER birincil, FINISH yalnız bu turun sonuçları isteği karşılıyorsa, eksik zorunlu bilgide ASK_CLARIFICATION). Registry açıklamaları Yürütme içindir (argüman ipuçları) ve 0.9'daki prompt'u ~825 token yapıyordu; Yasama her adımda çalıştığı için kısaltıldı. Şirket adaptasyonunda şirket araçları bu listeye bir satırla girer, liste ve veri birlikte değişir.
- Yürütme (~500 karakter): hiçbir araç adı geçmez; sözleşme istekteki TOOL_SCHEMA'dır, böylece aynı model şirket araçlarını da doldurur.
- Yargı (~600 karakter): gözleme dayanma, hata ve boş sonucu açıkça söyleme, boş gözlemde doğrudan cevap, ham JSON / iç ad yazmama. `RUNTIME_INFO` runtime'da ayarlıysa sona eklenir; eğitim verisinde yoktur (teacher'lı adımda yetenek soruları için eklenebilir).
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

Runtime tool'u: **CALCULATE** `{expression}` → observation `{result}` (§2.2, §3.1).

Veri gerçeği (`arac_kullanimi` clean_yyy_v2 tam tarama, 2026-10-19; Yürütme 136,169 · Yasama 222,934 · Yargı 86,765 calculator satırı):

| Kaynak | Source action | Yürütme satırı | Argümanlar | Observation |
|---|---|---|---|---|
| `tool-math` | `calc` | 84,908 (%62) | `{expression}` örn. `"15*0.6"` | `{result}` |
| `harsh8001-calc` + `yuyijiong-calc` | `calculator` | 39,636 (%29) | `{operation: add/subtract/multiply/divide, a, b}` | `{result}` |
| `glaive-v2-calc` | `calculate_*` (~175 ad) | 11,625 (%9) | alan formülleri (`bill_amount`, `tip_percentage`, ...) | alan adlı (`tip_amount`, `bmi`, ...) |

Eşleme:

```text
calc {expression}               → CALCULATE {expression}        (aynen)
calculator {add, a, b}          → CALCULATE {"a+b"}              subtract → "a-b", multiply → "a*b", divide → "a/b"
                                  negatif sayı parantezlenir: a=5, b=-3, subtract → "5-(-3)"
observation {result}            → aynen
```

Eşlemeden sonra her satır doğrulanır: ifade runtime CALCULATE ile yeniden hesaplanır ve kaynaktaki `result` ile (12 anlamlı basamakta) aynı olmalıdır; tutmayan satır atılır.

`tool-math` çok adımlıdır (aynı soruda 7+ hesaba kadar zincir; Yürütme step_index 0: 84,554, 1: 25,436, 2: 13,342 ...). Yasama tarafında her zincir FINISH ile biter: **86,765 `finish_after_tool` satırı** FINISH vs CONTINUE için en büyük kaynaktır.

**glaive `calculate_*` kararı:** canonical eğitime alınmaz (`unmapped_preserved` kalır). ~175 ayrı alan aracını runtime'a eklemek ontology'yi şişirir; argümanlardan ifade kurmak (BMI = kilo / boy², kredi taksiti...) formül varsayımı gerektirir ve observation'ı doğrulanamayan satır üretir. Gerekirse ileride yalnız formülü tartışmasız olanlar (`calculate_tip`, `calculate_discount`, `calculate_average`...) yukarıdaki doğrulamadan geçerek CALCULATE'e çevrilebilir.

Gürültü filtreleri (eğitimden önce):

- Rakamla yazılmış istekte operand istekte birebir geçmeli: `calculator` satırlarında 808 / 19,648 (%4) uyuşmazlık (`2247918 + 867` → `a: 22479418`).
- Yargı cevabı observation'daki sonucu (veya doğru yuvarlanmışını) içermeli; bozulmuş sayılar (`1.1833976833976836` → `1.183397668339766836`) atılır.
- `tool-math` Yargı cevaplarındaki `boxed9` gibi LaTeX artıkları temizlenir veya satır atılır.
- `tool-math` ifadeleri bazen ara akıl yürütme varsayar (30 kutu, "1 alana 1 bedava" → `15*0.6`); Yürütme bu satırlarda sorudan doğrudan çıkmayan sayıları öğrenir. Oranı ölçülmeli; gerekirse sorudaki sayılarla açıklanamayan ifadeler ayrı tutulur.

Calculator observation/result değerleri truncation ile değiştirilmemelidir.

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

### Kapsam analizi (2026-10-20, 17 action)

Dört kaynağın (`arac_kullanimi` clean_yyy_v2, `combined_v2`, Gemini distil, Hunter TR) tam taraması; action adları §2.1 alias'larıyla kanonikleştirildi, normalizasyon öncesi sayılar. Yargı "temiz": boş observation, İngilizce cevap, `<answer>` ve `boxed` içermeyen kayıt.

| Action | Yasama | Yürütme | Yargı (temiz) | Not |
|---|---:|---:|---:|---|
| FINISH | 711,300 | – | – | 678k `arac_kullanimi` |
| WEB_SEARCH | 599,282 | 599,282 | 591,884 | neredeyse tamamı `arac_kullanimi`; örnekleme ile dengelenmeli |
| CALCULATE | 124,546 | 124,546 | 41,767 | §6 |
| CODE_EXECUTE | 107,509 | 107,710 | 22 | ituperceptron Jupyter (pandas, `/home/user/input`); dar konu |
| SHELL_EXECUTE | 3,245 | 3,245 | 468 | Hunter |
| FILE_LIST | 2,778 | 1,600 | 1,017 | Hunter + Gemini |
| ASK_CLARIFICATION | 1,676 | – | – | BTX24 mobil |
| FILE_EDIT | 1,026 | 1,020 | 289 | |
| FILE_READ | 977 | 975 | 463 | |
| FILE_SEARCH | 365 | 365 | 336 | Gemini |
| FILE_DELETE / FILE_CREATE / FILE_MOVE | 186 / 116 / 78 | aynı | 174 / 116 / 76 | yalnız Gemini |
| RESPOND | **60** | – | 11 | |
| SEND_MAIL / DOWNLOAD / WEB_OPEN | 29 / 20 / 7 | aynı | 27 / 8 / 7 | |

- Çok turlu: Yasama'da geçmişli ~5.8k kayıt (çoğu FINISH), Yargı'da ~5k.
- Hata observation'ı: Yargı'da toplam 56 kayıt. Model "araç hata verdi" demeyi bu veriden öğrenemez.
- Yargı'da elenenler: `<answer>` 21,815 (ituperceptron), `boxed` 33,373 (`tool-math`), boş observation 10,685 (atasoglu), İngilizce cevap 519 (413'ü Hunter; Hunter Yargı'nın üçte biri).

**Yeterlilik:** Yasama büyük dört sınıfta (FINISH, WEB_SEARCH, CALCULATE, CODE_EXECUTE) yeterli; diğer ~10 sınıf, RESPOND ve çok turlu kayıtlar sentetik veri ister. Yürütme WEB_SEARCH ve CALCULATE'te yeterli; dosya araçlarında yüzlerce örnek var; WEB_OPEN / DOWNLOAD / SEND_MAIL sentetiksiz olmaz. Yargı en zayıf rol: web araması ve hesap dışında; hata durumları, dosya araçları, FILE_READ RAG biçimi (`partial: true`, `[satır N]`) ve "araç yapılandırılmamış" observation'ı tamamen sentetik üretilmelidir.

### Teacher'sız sentetik veri (0.8)

İlk adım teacher model gerektirmeyen, gerçek veriden programatik üretimdir (`datatools/synth.py`). Amaç üç rolün temel formatı öğrenmesi; rötuş (hata observation'ları, FILE_READ parçalı biçimi, "araç yapılandırılmamış", WEB_OPEN / DOWNLOAD / SEND_MAIL) sonraki teacher'lı adımdadır.

- **RESPOND.** Runtime RESPOND'u gözlemsiz Yargı ile cevaplar (`main.py`), bu yüzden iki rol veri ister.
  - Kaynaklar: `dpo.parquet` (`erenfazlioglu/turkish-instruct-reasoning-dpo-3.4m`), Human-Like-DPO-TR, `helpsteer3-tr` preference (yalnız `general` ve `stem`; `code` çevirileri bozuk; tercih edilen cevap `overall_preference < 0` ise `response1`).
  - Araç gerektirebilecek istekler kuralla elenir (`needs_tool`): aritmetik, hesap sözcükleri, güncel bilgi ve yakın yıllar, web / arama, URL ve e-posta adresi, dosya uzantısı, dosya / klasör / mail / indirme / komut sözcükleri.
  - **DPO çakışması:** Mercan DPO modeli `continuation_dpo_v1/train`'de bu üç kaynakla eğitildi (main_dpo 178,045 · helpsteer3 24,577 · human_like 8,146). Yasama yalnız isteği ve yeni bir etiketi (`RESPOND`) görür, bu yüzden üç kaynağın istekleri kullanılır (dpo %20 örneklenir). Yargı aynı cevaplarla yeniden SFT'lenirse ezber ve uzun deneme üslubu riski var, yeni bir şey öğrenmez; bu yüzden Yargı yalnız boş observation biçimini öğrenecek kadar veri alır: kısa (≤ 1200 karakter) dpo cevaplarından %2 örneklem, **yalnız train**; ve DPO'da tutulmuş `helpsteer3-tr` validation (`md5` ile %50 train, %25 validation, %25 test). Yargı RESPOND eval'i yalnız bu görülmemiş kayıtlardan gelir. Human-Like cevapları kişisel deneyim iddia ettiği için hiç kullanılmaz.
- **Geçmiş karıştırma.** Mevcut kayıtlara alakasız bir önceki tur eklenir, etiket değişmez. Bu, Yasama'nın önceki action'ı kopyalamasını ve Yargı'nın geçmişi cevap diye basmasını doğrudan eğitir.
  - Önceki tur havuzu: kısa (≤ 1200 karakter) dpo sohbet turları (%3) ve gerçek araç turları (`arac_kullanimi` calculator istek + Yargı cevabı, %0.6); v5'te 4,379 + 3,932 tur.
  - Oranlar: büyük sınıflar (WEB_SEARCH, CALCULATE, CODE_EXECUTE, FINISH, FILE_CREATE) %0.4, açık araçlar %10, diğer tüm sınıflar %100. Yasama ve Yürütme yalnız ilk adım kayıtlarını, Yargı tümünü karıştırır.
  - Kopya `source_id#hist` alır ve orijinalinin split'inde kalır (test kaydı train'e sızmaz).

### Format uyumu (2026-10-20)

- Observation dört alanı her kaynakta var; 734k observation'da fazladan `tool_call_id` (atılır).
- STATE: `completed_actions` çoğunlukla ad listesi; 746 kayıtta obje (ada çevrilir); 236k eski kayıtta `recent_observations` yok (`[last_observation]` ile kurulur).
- Yürütme argümanları registry'ye göre: WEB_SEARCH 599,277 / 599,282 uyumlu; CALCULATE 84,908 uyumlu + 39,636 dönüştürülebilir; CODE_EXECUTE 107,688 `{answer, code}`; Hunter `file_path` / `dir_path`; Gemini FILE_SEARCH `pattern`. Dönüşümler §2.2 tablosunda.
- Yargı observation data'sı §3.1'e göre: CALCULATE 124,528 uyumlu. **WEB_SEARCH 597,730 kayıtta `{"result": "[1] url\nmetin\n\n[2] ..."}` tek metin** → `[n] url` bloklarından `results: [{url, snippet}]` ayrıştırılır, `query` aynı adımın Yürütme argümanından alınır. FILE_LIST / FILE_READ / FILE_SEARCH'te Hunter `{success, result, tool, source}` kabuğu ve Gemini `files` / `results` / `found_files` adları `entries` / `content` / `matches`'e çevrilir.

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

**Güncel sürüm v5, temel format eğitimi için kullanılabilir; henüz ACTIVE değil** (teacher'lı sentetik veri ve örnekleme oranları belirlenene kadar):

```text
/arf/scratch/egitimg16/nedolm_0p8b/tooluse_inspect/yasama-yurutme-yargi/shared_data/v5/
├── manifest.json              aşağıdaki alanlar + kaynak sha256, sayılar, ret sebepleri
├── contracts/tool_registry.json
├── rejects/<rol>.samples.jsonl  her ret sebebinden örnek
└── yasama|yurutme|yargi/{train,validation,test}.jsonl.gz
```

Üretim: `python -m datatools.build_shared_data --root <tooluse_inspect> --out <shared_data/vN>` (SoP repo; TRUBA'da CPU `debug` kuyruğu, `apps/truba-ai/cpu-2024.0`). Kayıt:

```text
{"id", "source", "source_dataset", "split", "action", "step_index", "has_history",
 ["error_observations"], "messages": [request turn(s)..., {"role": "assistant", "content": <target>}]}
```

v5 sonucu (2026-10-21, kod `a8b2cffe`, schema 4, 12.6 dk; v4 = + teacher'sız RESPOND ve geçmiş karıştırma, v5 = Yargı RESPOND DPO çakışmasına göre sınırlandı, §7):

| Rol | train | validation | test | Toplam | v3 | v4 |
|---|---:|---:|---:|---:|---:|---:|
| Yasama | 1,442,095 | 33,717 | 34,690 | 1,510,502 | 1,451,368 | 1,510,502 |
| Yürütme | 811,608 | 20,859 | 21,706 | 854,173 | 847,895 | 854,173 |
| Yargı | 664,291 | 14,682 | 14,952 | 693,925 | 684,985 | 735,872 |

- Yasama RESPOND 60 → **52,832** (sohbet kaynakları 52,734; tekrar olduğu için 8,612 istek elendi). Geçmişli kayıt 43 → **12,454** (geçmiş karıştırma 6,400; ASK_CLARIFICATION 1,661, FILE_LIST 1,161 gibi nadir sınıflar ikiye katlandı).
- Yürütme geçmişli kayıt 4,541 → **10,819** (geçmiş karıştırma 6,278).
- Yargı RESPOND 0 → **3,750**: train 3,262 (kısa dpo 2,811 + HelpSteer3 validation 451), validation 226 ve test 262 yalnız HelpSteer3 validation'dan. v4'te kısıtsız sohbet cevaplarıyla 45,697 idi. Geçmişli kayıt 4,546 → **10,210** (geçmiş karıştırma 5,190); hata observation'lı 84.
- Geçmiş karıştırma Yargı retleri: İngilizce cevap 366, uzunluk 135; orijinal kayıtlarla aynı sebepler.
- Hâlâ boş: WEB_OPEN 11, DOWNLOAD 32, SEND_MAIL 38 (Yasama) ve Yargı hata observation'ları; teacher'lı adıma kaldı (§14).

v3 sonucu (2026-10-20, kod `71ce9b03`, 12.8 dk; v1 = ilk build, v2 = + `write_file` koşullu alias, v3 = + açık araçlar ve uzunluk sınırı):

| Rol | train | validation | test | Toplam | v1 | v2 |
|---|---:|---:|---:|---:|---:|---:|
| Yasama | 1,385,491 | 32,439 | 33,438 | 1,451,368 | 1,437,369 | 1,451,368 |
| Yürütme | 805,672 | 20,675 | 21,548 | 847,895 | 788,235 | 797,680 |
| Yargı | 656,096 | 14,325 | 14,564 | 684,985 | 672,038 | 672,690 |

- Yasama: FINISH 650,459 · WEB_SEARCH 554,292 · CALCULATE 121,717 · CODE_EXECUTE 107,107 · FILE_CREATE 10,933 · FILE_LIST 2,204 · ASK_CLARIFICATION 1,675 · SHELL_EXECUTE 1,374 · FILE_READ 543 · FILE_EDIT 400 · FILE_SEARCH 293 · FILE_DELETE 185 · FILE_MOVE 71 · RESPOND 60 · SEND_MAIL 28 · DOWNLOAD 20 · WEB_OPEN 7. Geçmişli 43.
- Yürütme: WEB_SEARCH 554,607 · CALCULATE 123,038 · CODE_EXECUTE 107,693 · **açık araç 46,632 (15,863 farklı araç; combined_v2 26,774, Hunter 10,674, glaive 9,184)** · FILE_CREATE 7,410 · SHELL_EXECUTE 3,229 · FILE_LIST 2,756 · FILE_EDIT 972 · FILE_READ 942 · FILE_SEARCH 317 · FILE_DELETE 182 · FILE_MOVE 73 · SEND_MAIL 28 · DOWNLOAD 9 · WEB_OPEN 7. Geçmişli 4,541.
- Yargı: WEB_SEARCH 591,476 · CALCULATE 57,755 · CODE_EXECUTE 21,694 · **açık araç 12,375 (glaive 8,010, combined_v2 4,365)** · dosya araçları ve karışık zincirler ~1.7k · hata observation'lı 51 · geçmişli 4,546.
- Yürütme açık araç retleri: şema yok 33,552 (ituperceptron, BTX24), argüman şemaya uymuyor ~1.1k (çoğu tip: object yerine str/list), liste biçimli şema 201 (bilalabic). Uzunluk sınırı: Yürütme 485, Yargı 253 kayıt; FILE_CREATE içerik sınırı 3,647.
- Başlıca retler: tekrar (Yasama 82k, Yürütme 40k), ontology dışı action (glaive `calculate_*`, combined_v2 kaynak araçları, Hunter `write_file`), Hunter `write_file` içeren adım zinciri (`state_unmapped_action`: Yasama 7,001 — Hunter dosya kayıtlarının çoğu bu yüzden düşer), `web_result_unparsed` ~6k (URL'siz "1. Belge … Puan" arama biçimi), Yargı `boxed` 9,164 + yalnız artıktan oluşan cevap 5,372 + sayı bozulması 2,839, boş observation 10,685, calculator operand uyuşmazlığı 809.

`messages` rollerin runtime'daki `request_message` / `request_messages` fonksiyonlarıyla üretilir (§4); system prompt ve few-shot turları eklenmez. Provenance alanları mesajlara girmez. Split: `combined_v2` kendi split'ini korur, diğerleri `md5(source:source_id)` ile (test %2, validation %2), böylece bir konuşmanın tüm adımları ve rolleri aynı split'te kalır. Aynı `messages` rol içinde bir kez yazılır.

## 9.1 Eğitim karışımı (`mix/`, 0.9)

shared_data sözleşmeye uyan her kaydı tutar; eğitim için `datatools/build_mix.py` bir karışım üretir:

```text
python -m datatools.build_mix --shared shared_data/v5 --out mix/v1
mix/v1/<rol>/{train,validation,test}.jsonl.gz   kayıt = shared_data kaydı, messages[0] = rolün system mesajı
mix/v1/manifest.json                            katman başına mevcut/alınan, kalite retleri, cap'ler, prompt sha256
```

- Küçük sınıflar ve diğer kaynaklar **bütün** alınır. Büyük katmanlar önce kalite filtresinden geçer, sonra kayıt id'sinin md5'ine göre en küçük N tanesi alınır (deterministik; yeniden üretimde aynı kayıtlar).
- Kalite filtreleri: ms-marco isteği 20–200 karakter, büyük harf / rakamla başlayan ve soru biçiminde ("?" veya soru sözcüğü / eki) olmalı ("ms bir nörodejeneratif hastalık" gibi anahtar kelime parçaları düşer); aynı epizotların FINISH kayıtlarına da uygulanır. Yargı web cevabı 30–800 karakter, cümle sonu noktalamalı ve içerik sözcüklerinin (5 harflik kök) en az yarısı arama sonuçlarında geçmeli.
- Validation / test katmanları train cap'inin %5'i ile sınırlanır (eval, kaynak çarpıklığını değil karışımı ölçer).

- Eğitim uzunluğu (mix 2): system mesajı dahil `nedolm.chat_format` serialization'ında **8,192 NDSRF004 token**'ı aşan kayıt alınmaz (`--max-tokens`, `--vocab`); cap'li katmanlar boşalan yeri aynı katmandan doldurur.

**mix/v3** (2026-10-24, kod `f9b30236`, shared_data v6): mix/v2 ile aynı kurallar; Yasama STATE'i son üç adımın `data_preview`'ını taşır (§3.3). Yasama train 193,732 kayıt / 118.9 M token (+%5); Yürütme ve Yargı v2 ile aynı.

**mix/v2** (2026-10-21, kod `71df41db`, shared_data v5; v1'den farkı kısa Yasama prompt'u ve 8,192 sınırı). Token sayıları eğitimdeki biçimle (rol başlıkları, kontrol token'ları, hedef sonrası EOS) ölçüldü (`mix/v2/TOKEN_COUNTS.json`):

| Rol | train kayıt | Toplam token / epoch | Hedef (loss) token | System payı | Kayıt başı ort. / p95 | Uzunluktan elenen |
|---|---:|---:|---:|---:|---:|---:|
| Yasama | 193,670 | 112.8 M | 2.3 M | 74.4 M | 582 / 887 | 0 |
| Yürütme | 120,197 | 115.3 M | 29.6 M | 17.1 M | 960 / 2,640 | 31 |
| Yargı | 81,374 | 58.1 M | 3.8 M | 13.9 M | 714 / 1,698 | 6 (+1 validation) |
| Toplam | 395,241 | 286.2 M | 35.7 M | | | |

v1'de Yasama 198.3 M token'dı (%81'i system prompt). Validation / test: Yasama 8,349 / 8,420, Yürütme 5,250 / 5,412, Yargı 3,780 / 3,856.

mix/v1 (2026-10-21, kod `14d7a8c0`, shared_data v5):

| Rol | train | validation | test | Cap'li katmanlar (alınan / filtre sonrası mevcut) |
|---|---:|---:|---:|---|
| Yasama | 193,670 | 8,349 | 8,420 | ms-marco WEB_SEARCH 40k / 226k · FINISH web sonrası 25k / 226k, hesap sonrası 15k / 71k, kod sonrası 8k / 20k · CALCULATE 30k / 117k · CODE_EXECUTE 10k / 97k · sohbet RESPOND 40k / 51k |
| Yürütme | 120,218 | 5,250 | 5,412 | ms-marco WEB_SEARCH 15k / 226k (sorgu %99.6 isteğin kopyası) · CALCULATE 30k / 118k · CODE_EXECUTE 10k / 97k |
| Yargı | 81,374 | 3,780 | 3,856 | ms-marco WEB_SEARCH 30k / 179k · CALCULATE 20k / 55k · CODE_EXECUTE 10k / 20k |

- Kalite filtresi ms-marco isteklerinin %57'sini, Yargı web cevaplarının %68'ini eler.
- Yasama train: FINISH 49,710 · WEB_SEARCH 43,624 · RESPOND 40,056 · CALCULATE 30,278 · FILE_CREATE 10,224 · CODE_EXECUTE 10,095 · diğer 13 sınıf ~9.7k. Yürütme train: açık araç 44,685 (tamamı) · CALCULATE 30,000 · WEB_SEARCH 18,626 · CODE_EXECUTE 10,095 · dosya / shell ~16.8k. Yargı train: WEB_SEARCH 32,241 · CALCULATE 20,233 · açık araç 12,592 · CODE_EXECUTE 10,072 · RESPOND 3,262 · dosya zincirleri ~3k.
- Bilinen sınır: WEB_OPEN / DOWNLOAD / SEND_MAIL ve hata observation'ları hâlâ onlarca kayıt; teacher'lı adımda (§14).
- Not: ms-marco bilgi soruları WEB_SEARCH, dpo açıklama / tavsiye soruları RESPOND etiketlidir. Sınır çoğunlukla doğru (somut varlık / sayı / tarih → arama), ama model üslup ipucuyla (kısa sorgu → arama) ayırmayı öğrenebilir; eval'de iki yönlü karışma ayrıca ölçülmeli.

## Base checkpoint (0.9)

Üç rol **Mercan-0.8B-DPO** üzerine eğitilir: `runs/nedolm-0p8b-continuation-dpo-v1-step1436-2xh200-prod-20260928/checkpoints/step_00007895.pt` (continuation SFT → DPO; DPO tek epoch, 7,895 adım × 32 çift, β 0.1, lr 5e-7). Gerekçe: en iyi Türkçe sohbet davranışı ve aynı tokenizer / sohbet şablonu (`sistem` / `kullanici` / `asistan`); önceki Yargı denemeleri de bu checkpoint'ten yapıldı. Sonraki GRPO/RLVR checkpoint'i çoktan seçmeli muhakemeye optimize edildiği için kullanılmaz; `Mercan-0.8B-SFT` hiçbir yerde kullanılmaz. Unutma riskine karşı: Yargı karışımında DPO'da görülmemiş / kısa sohbet cevapları (RESPOND) vardır ve her rolün eğitim sonrası genel Türkçe sohbet ile rol eval'i (§12) DPO checkpoint'iyle karşılaştırılır.

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
- **FILE_READ sorgu temizliği:** Uzun dosyada Yürütme'nin `query`'sine kullanıcının isteği eklenir; Yürütme'nin sorguyu İngilizceye çevirdiği veya uydurduğu görüldü. Eğitilmiş Yürütme ile yeniden ölçülmeli.
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
CALCULATE vs CODE_EXECUTE / RESPOND
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
CALCULATE: sonuç doğruluğu (ifade runtime'da hesaplanıp beklenen sonuçla karşılaştırılır; ifade metni birebir aranmaz), yazıyla yazılmış ve ondalık virgüllü sayılar
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

## İlk rol eğitimleri ve ölçümler (0.11, 2026-10-22 / 24)

Eğitim: `training/train_role.py` (TRUBA, tek H200, `training/train_role.slurm`). DPO `step_00007895.pt`'den tam parametre; kayıtlar `nedolm.chat_format` biçiminde, EOS'ta bölünen 8,193 token'lık satırlara paketlenir (FlashAttention varlen, pozisyonlar konuşma başına sıfırlanır); loss yalnız son asistan mesajı + `<|mesaj_bitir|>`. AdamW (0.9, 0.95), wd 0, clip 1.0, %3 warmup, cosine → peak'in %10'u; adım = 3 × 4 satır ≈ 98k token; ~68–74k token/s, tepe bellek 74.5 GiB (H100'e sığmaz). Ölçüm: `training/eval_role.py` + `training/eval_metrics.py` (greedy üretim, test split).

| Rol | Veri | LR / epoch | Süre | Checkpoint (`runs/…/checkpoints/`) |
|---|---|---|---:|---|
| Yasama | mix/v3 | 1e-5 / 2 | 60 dk | `mercan-yasama-mixv3-dpo7895-20261024/…/epoch_02_step_002532.pt` |
| Yürütme | mix/v2 | 1e-5 / 2 | 68 dk | `mercan-yurutme-mixv2-dpo7895-20261022/…/epoch_02_step_002601.pt` |
| Yargı | mix/v2 | 5e-6 / 2 | 32 dk | `mercan-yargi-mixv2-dpo7895-20261022/…/epoch_02_step_001272.pt` (epoch 2 val loss 1.057 ≈ epoch 1) |

- **Yasama** test doğruluğu: mix/v2 (STATE'te yalnız sonuç) %93.55 → mix/v3 (son üç adıma `data_preview`, §3.3) **%95.34**; geçersiz çıktı 0. Hatalar 543 → 392; CODE_EXECUTE → FINISH 191 → 35, recall CODE_EXECUTE %61.9 → %92.8. Kalan hataların %56'sı hesap zincirlerinde FINISH ↔ CALCULATE (144 + 76; önizleme bunu değiştirmedi: sonucun soruyu karşılayıp karşılamadığı çok adımlı akıl yürütme ister). Zayıf: SHELL_EXECUTE %39, FILE_EDIT %47, FILE_READ %56 (test'te 18–28 örnek; teacher'lı veri bekler).
- **Yürütme** (action başına 40 örnek, 384 kayıt): JSON + şema geçerli %83.3; WEB_SEARCH %97.5 doğru, FILE_READ / FILE_SEARCH / FILE_LIST / CALCULATE çıktılarının %100'ü şemaya uygun; açık araçlar şema %90, birebir %45; CODE_EXECUTE kodlarının %100'ü Python olarak derlenir. CALCULATE anlamsal doğruluk (ifade sonucu eşit) %42.5: kelime problemlerinde yanlış ara adım. Uzun içerikte greedy tekrar döngüsü: 3,072 token sınırına takılan 64 çıktının 60'ı FILE_CREATE / FILE_EDIT / SHELL_EXECUTE (Hunter; hedef istekten çıkarılamaz).
- **Yargı** (action başına 30 örnek, 219 kayıt), aynı kayıtlarda DPO base ile: JSON / iç biçim sızıntısı %38 → **%0**, gözlemde olmayan sayı %17.5 → %10.6, İngilizce %5.9 → %2.3, referansla kelime F1 0.16 → 0.53, uzunluk / referans 6.6× → ~1×. 12 genel sohbet sorusunda DPO base Yargı biçiminde 11 kez sahte `<tool_observations>` üretir, Yargı hepsini düzgün cevaplar; bilgi hataları sürer (0.8B sınırı). Test örnekleminde hata observation'ı yok: "hatayı kabul etme" ölçülemedi (teacher'lı adım).
- Açık: Yürütme ve Yargı için runtime'da tekrar cezası / `max_tokens`; FINISH ↔ CALCULATE için hesap zincirlerinin incelenmesi; `.mercan` export ve uçtan uca CLI denemesi.

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
- teacher'lı sentetik veri (§7): teacher seçimi (API / TRUBA'da açık model); Yargı hata observation'ları, FILE_READ parçalı biçimi, yapılandırılmış sonuçlar, "araç yapılandırılmamış"; WEB_OPEN / DOWNLOAD / SEND_MAIL; Yargı RESPOND cevaplarının teacher ile yeniden yazılması
- şeması olmayan kaynaklar (ituperceptron Jupyter dışı ~20k, BTX24 mobil ~15k): şema yazılırsa Yürütme'ye açık araç olarak girer
- Yasama'nın action listesini girdi olarak alması (şirket adaptasyonunu ucuzlatır; verideki `available_tools` gibi)
- exact active source listesi
- mix cap'lerinin ilk eğitim sonuçlarına göre ayarlanması (§9.1)
- Yasama'nın fiziksel model/checkpoint standardı
- Yürütme ve Yargı tek checkpoint mi iki checkpoint mi olacak
- tokenizer/chat-template kesin sürümü
- max sequence length (runtime varsayılanı 8192 olarak kararlaştırıldı — §3.2; training max length açık)
- ortak experiment/run directory standardı
- checkpoint naming convention
- evaluator script contract'ları
- runtime service/process topolojisi
- `tool-math` ifadelerinde sorudan çıkmayan ara sonuç oranı ve bu satırların kullanımı (§6)
```

Bu konularda karar verilene kadar mevcut davranış "resmi standart" gibi varsayılmamalıdır.

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

## 0.11 — 2026-10-24

- §3.3: Yasama STATE'inde son üç observation'a `data_preview` (data'nın ilk 200 karakteri). Gerekçe: rol eval'inde Yasama hatalarının %85'i çok adımlı zincirlerde devam / bitir kararıydı.
- §9.1: mix/v3 (shared_data v6, schema 5).
- §12: İlk rol eğitimleri (DPO base, mix/v2–v3) ve üretimli eval sonuçları; Yasama test %95.34, Yürütme şema geçerliliği %83.3, Yargı JSON sızıntısı %0.

## 0.10 — 2026-10-21

- §4: Yasama eğitim prompt'u kısaltıldı: action başına tek satırlık özet (1,200 karakter, ~384 token; önce ~825 token).
- §9.1: mix/v2: kısa Yasama prompt'u, 8,192 token eğitim uzunluğu; NDSRF004 ile ölçülmüş token sayıları (toplam 286 M token / epoch, 35.7 M hedef token).

## 0.9 — 2026-10-21

- §4: Rol başına eğitim system prompt'u; runtime `prompt_style: "trained"` ile aynısını gönderir, few-shot turlarını eklemez.
- §9.1: Eğitim karışımı `mix/v1` (kalite filtresi + deterministik örnekleme; Yasama 193,670 / Yürütme 120,218 / Yargı 81,374 train).
- §9: Base checkpoint kararı: Mercan-0.8B-DPO `step_00007895.pt`.
- §14: System prompt ve örnekleme oranı maddeleri kapandı.

## 0.8 — 2026-10-21

- §7: Teacher'sız sentetik veri: RESPOND (sohbet kaynakları, `needs_tool` filtresi) ve geçmiş karıştırma. DPO çakışması kuralı: Yasama DPO'da görülmüş isteklerin hepsini kullanır, Yargı yalnız ~3k biçim örneği (train) alır ve RESPOND eval'i DPO'da tutulmuş HelpSteer3 validation'dan gelir.
- §9: shared_data v4 ve v5; güncel v5 sayıları (Yasama RESPOND 52,832, Yargı RESPOND 3,750, geçmişli kayıtlar Yasama 12,454 / Yürütme 10,819 / Yargı 10,210).
- §14: Sentetik veri maddesi teacher'lı adıma daraltıldı.

## 0.7 — 2026-10-20

- §2.1: `write_file` → FILE_CREATE koşullu alias (yalnız yeni dosya yazan Hunter adımları; ölçüm ve kural).
- §2.2: Açık araçlar: Yürütme ve Yargı verisi registry dışı araçları kaynak şemasıyla tutar (hedef: şirkete özel araçlar). FILE_CREATE içeriği ≤ 7000 karakter; her rolde kayıt ≤ 16,000 karakter.
- §9: shared_data v2 ve v3; güncel v3 sayıları (Yürütme 46,632 açık araç kaydı, 15,863 farklı araç).
- §14: Hunter `write_file` kararı kapandı; şemasız kaynaklar ve Yasama'ya girdi olarak action listesi açık.

## 0.6 — 2026-10-20

- §2.2: Kaynak argüman adlarının registry'ye dönüşüm tablosu (`datatools/normalize.py`).
- §3.1: WEB_SEARCH `title` isteğe bağlı (veride yok).
- §7: 17 action'lık rol kapsamı, format uyumu ve kalite bulguları (tam tarama).
- §9: `shared_data/v1` üretildi (henüz ACTIVE değil): `datatools/build_shared_data.py`, kayıt biçimi, split ve tekrar politikası, v1 sayıları ve ret sebepleri. Mesajlar rollerin runtime request builder'larıyla üretilir; system prompt eklenmez.
- §14: shared_data dosya biçimi (JSONL.GZ) kapandı; eğitim system prompt'u, sentetik veri ve Hunter `write_file` açık.

## 0.5 — 2026-10-19

- §2.2, §3.1, §6: CALCULATE şeması `{expression}` oldu (verinin %62'si `tool-math` `calc {expression}`); `calculator {operation, a, b}` satırları kayıpsız ifadeye çevrilir. Hesap `ethosoftlib.math.evaluate` ile (eval yok).
- §2.2: Fiziksel tool registry `runtime/tool_registry.json` (format 1); koddan üretilir, test eskimesini yakalar. §14'teki açık karar kapandı.
- §6: Calculator verisinin tam profili, eşleme ve doğrulama kuralı, gürültü filtreleri. glaive `calculate_*` canonical eğitime alınmaz.
- §12: CALCULATE için sonuç doğruluğu ölçütü.
- Ölçüm notu (genel 0.8B Yürütme, few-shot ile, aynı sorular): `{operation, a, b}` şeması 3/8; `{expression}` şeması ilk 8 soruda 6/8, 10 soruda 7/10 (doğru işlem seçimi, yazıyla yazılmış ondalıklar ve `240*(1-0.15)` gibi bileşik ifadeler doğru). Kalan hatalar: karekökte yalnız `625`, çarpma yerine bölme, çok adımlı soruda yalnız ilk adım.

## 0.4 — 2026-10-19

- §2.1, §2.2: **CALCULATE** 17. canonical action. Şema calculator verisinin `{operation, a, b}` şeması + `power` / `sqrt`. `calculator → CALCULATE` güvenli alias; glaive `calculate_*` alan araçları map edilmez.
- §2.2, §3.1: FILE_READ `max_chars` kaldırıldı, `query` eklendi. Uzun dosya ve PDF `ethosoftlib.rag` (BM25) ile soruya göre okunur; ≤ 7000 karakter dosya tam döner. `truncated` alanı kalktı.
- §3.1: CALCULATE observation `{result}`; sonuç biçimlendirme ve hata kuralları.
- §3.2: Büyük dosyalar için `offset` yerine RAG.
- §6: Calculator politikası yeniden yazıldı: veri eşlemesi, `finish_after_tool` kaynağı, gürültü filtreleri.
- §12: CALCULATE için Yasama/Yürütme ölçütleri.
- §14: "tool listesinin yeniden düzenlenmesi" kapandı; glaive alan araçları açık karar olarak kaldı. Tool registry'nin fiziksel şeması hâlâ açık.
- Ölçüm notu (genel 0.8B checkpoint, few-shot ile): Yasama 5 hesaplama sorusunda CALCULATE'i 0/5 seçti (CODE_EXECUTE, SHELL_EXECUTE veya `SQRT` gibi uydurma adlar); üç örnekli varyant 2/5'e çıktı ama FILE_LIST'i bozdu. Yürütme CALCULATE argümanlarında 3/8 (sayılar çoğunlukla doğru, `operation` çoğunlukla `power`). Her ikisi de calculator verisiyle eğitilecek. Hesap dışı Yasama kararları değişmedi (7 soruda 5/7, eskisiyle aynı).

## 0.3 — 2026-10-19

- §3.2: Observation'lar kesilmez. Yasama observation'ların yalnız action/status/error'unu görür; Yürütme ve Yargı tam görür. Serialize edilen STATE'te tekrar eden `last_observation` bir kez yazılır.
- §3.2: libmercan rolleri için varsayılan context 8192; büyük dosyalar parça parça okunacak (FILE_READ `offset`, tool listesi revizyonu).
- §4: Yasama'ya konuşma geçmişi yalnız turun ilk adımında verilir; sonraki adımlarda yalnız adım zinciri.
- §14: "observation kısaltma" ve "Yasama geçmiş biçimi" açık kararları kapatıldı.
- Ölçüm notu: genel 0.8B checkpoint, başarılı bir tool sonrasında FINISH'i hiçbir STATE görünümünde güvenilir seçemiyor (5 senaryoda 0–1/5; kısaltmalı eski görünüm 0/5). FINISH vs CONTINUE eğitimle çözülecek; o zamana kadar tekrar eden action+argüman koruması turu Yargı'ya bağlar.

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
