"""
محاكاة التعلم الاتحادي على قاعدة بيانات أمراض النبات (PlantVillage)
مبني على نفس معمارية CNN المخصصة (CBAM + Inverted Residual) ونفس منطق تحميل البيانات الأصلي.

الإصدار الحالي: كل عميل (ومرحلة التقييم) يُشغَّل في عملية نظام تشغيل منفصلة تمامًا
(multiprocessing بوضع spawn) تُنهى فور انتهائها. هذا يحل مشكلة تراكم ذاكرة
TensorFlow/XLA عبر الجولات بشكل جذري: نظام التشغيل يستعيد كل الذاكرة (RAM+GPU)
بمجرد انتهاء كل عملية فرعية، بغض النظر عن مصدر التسرب الداخلي.

كما يسجّل هذا الإصدار مقاييس كفاءة الموارد لكل جولة (RAM، زمن التدريب المحلي،
زمن الاستدلال، حجم البيانات المُرسلة/المُستلمة) في ملف CSV — هذه المقاييس هي
محور "كفاءة النشر لصغار المزارعين" الذي حددناه كمساهمة بحثية أصيلة.

قبل التشغيل:
  pip install tensorflow scikit-learn pandas

طريقة التشغيل:
  python federated_plantvillage.py
"""

import os
import gc
import time
import json
import psutil
import numpy as np
import multiprocessing as mp

# -------------------- الإعدادات القابلة للتعديل --------------------
DATA_PATH = '/content/gdrive/MyDrive/color'   # عدّل هذا المسار حسب موقع بياناتك

# -------------------- مسارات منفصلة تمامًا عن تجربة IID — لا تداخل ولا استئناف خاطئ --------------------
CKPT_DIR = '/content/gdrive/MyDrive/fl_ckpt_dirichlet_a05'
RESULTS_CSV = '/content/gdrive/MyDrive/fl_results_dirichlet_a05.csv'
MODELS_DIR = '/content/gdrive/MyDrive/fl_models_dirichlet_a05'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

NUM_CLIENTS = 5
TARGET_ROUNDS = 30           # نفس عدد جولات IID لضمان مقارنة عادلة (نفس عدد التكرارات بالضبط)
LOCAL_EPOCHS = 3
MAX_IMAGES_PER_CLIENT = 3000   # نفس القيمة المستخدمة في IID — شرط ضروري للمقارنة العادلة
SNAPSHOT_EVERY = 5

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------- تحميل مسارات الصور (نفس منطقك الأصلي) --------------------
def load_image_paths_and_labels(data_path):
    paths, labels = [], []
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                paths.append(os.path.join(class_path, img_file))
                labels.append(class_name)
    return np.array(paths), np.array(labels)


def partition_iid(paths, labels, num_clients, seed=42):
    """توزيع متجانس: كل عميل يحصل على نسبة متقاربة من كل صنف (خط الأساس)."""
    rng = np.random.default_rng(seed)
    indices_by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in indices_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    client_data = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        client_data.append((paths[idxs], labels[idxs]))
    return client_data


def partition_dirichlet(paths, labels, num_clients, alpha=0.5, seed=42):
    """توزيع غير متجانس واقعي (تخصص كل مزرعة) — Hsu et al., 2019."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        cutoffs = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cutoffs)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    client_data = []
    for idxs in client_indices:
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        client_data.append((paths[idxs], labels[idxs]))
    return client_data


# -------------------- تُحمَّل هذه القيم مرة واحدة في كل عملية (أب أو فرعية) --------------------
print("جارٍ تحميل مسارات الصور...")
image_paths, labels_raw = load_image_paths_and_labels(DATA_PATH)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels_encoded = le.fit_transform(labels_raw)
num_classes = len(le.classes_)
print(f"عدد الأصناف: {num_classes} | عدد الصور الكلي: {len(image_paths)}")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    image_paths, labels_encoded, test_size=0.15, random_state=SEED, stratify=labels_encoded
)

DIRICHLET_ALPHA = 0.5  # درجة عدم التجانس: كلما قلّ = تخصص أكبر (محاكاة واقع مزرعة متخصصة)
print(f"جارٍ تقسيم البيانات على {NUM_CLIENTS} عميل (توزيع Dirichlet غير متجانس، alpha={DIRICHLET_ALPHA})...")
client_partitions = partition_dirichlet(X_trainval, y_trainval, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED)
for i, (p, l) in enumerate(client_partitions):
    n_unique = len(np.unique(l))
    dominant_class_share = np.max(np.bincount(l)) / len(l) if len(l) > 0 else 0
    print(f"  عميل {i}: {len(p)} صورة عبر {n_unique} صنفًا من أصل {num_classes} "
          f"| نسبة الصنف المهيمن: {dominant_class_share*100:.1f}%")

# للتبديل إلى السيناريو الواقعي (تخصص المزارع)، استبدل السطر أعلاه بـ:
# client_partitions = partition_dirichlet(X_trainval, y_trainval, NUM_CLIENTS, alpha=0.3, seed=SEED)


# -------------------- دوال تُستخدم فقط داخل العمليات الفرعية (تستورد tensorflow محليًا) --------------------
def _build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    from tensorflow.keras.optimizers import Adam

    def CBAMLiteBlock(x, ratio=8):
        filters = x.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(x)
        dense = layers.Dense(filters // ratio, activation='relu')(avg_pool)
        dense = layers.Dense(filters, activation='sigmoid')(dense)
        channel_att = layers.Multiply()([x, layers.Reshape([1, 1, filters])(dense)])
        spatial_avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(channel_att)
        spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_avg)
        spatial_att = layers.Multiply()([channel_att, spatial])
        return spatial_att

    def InvertedResidualBlock(x, filters, expansion=4, strides=1, use_cbam=True):
        in_channels = x.shape[-1]
        expanded = layers.Conv2D(in_channels * expansion, 1, padding='same', use_bias=False)(x)
        expanded = layers.BatchNormalization()(expanded)
        expanded = layers.Activation('swish')(expanded)
        dw = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(expanded)
        projected = layers.Conv2D(filters, 1, padding='same', use_bias=False)(dw)
        projected = layers.BatchNormalization()(projected)
        if use_cbam:
            projected = CBAMLiteBlock(projected)
        if strides == 1 and in_channels == filters:
            return layers.Add()([x, projected])
        return projected

    inputs = Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = InvertedResidualBlock(x, 32)
    x = InvertedResidualBlock(x, 64, strides=2)
    x = InvertedResidualBlock(x, 64)
    x = InvertedResidualBlock(x, 128, strides=2)
    x = InvertedResidualBlock(x, 128)
    x = InvertedResidualBlock(x, 256, strides=2)
    x = InvertedResidualBlock(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])
    return m


def _make_dataset(paths, labels, batch_size, shuffle=False, augment=False):
    import tensorflow as tf

    def decode_and_preprocess(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=SEED)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _save_weights(path, weights):
    np.savez(path, **{f"w{i}": w for i, w in enumerate(weights)})


def _load_weights(path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda k: int(k[1:]))
    return [data[k] for k in keys]


def _save_full_keras_model_target(weights_path, out_model_path):
    """يبني النموذج الكامل من الأوزان ويحفظه بصيغة .keras — قابل للفتح مباشرة لاحقًا
    بـ tf.keras.models.load_model() دون الحاجة لإعادة كتابة المعمارية يدويًا."""
    weights = _load_weights(weights_path)
    model = _build_model()
    model.set_weights(weights)
    model.save(out_model_path)


# -------------------- الأهداف التي تُنفَّذ كل واحدة منها في عملية منفصلة تمامًا --------------------
def _init_weights_target(out_weights_path):
    m = _build_model()
    _save_weights(out_weights_path, m.get_weights())


def _client_target(cid, global_weights_path, out_weights_path, out_info_path):
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    global_weights = _load_weights(global_weights_path)
    paths, labels_c = client_partitions[cid]
    if MAX_IMAGES_PER_CLIENT is not None and len(paths) > MAX_IMAGES_PER_CLIENT:
        rng = np.random.default_rng(SEED + cid)
        idx = rng.choice(len(paths), size=MAX_IMAGES_PER_CLIENT, replace=False)
        paths, labels_c = paths[idx], labels_c[idx]

    model = _build_model()
    model.set_weights(global_weights)

    ds = _make_dataset(paths, labels_c, BATCH_SIZE, shuffle=True, augment=True)
    t0 = time.time()
    model.fit(ds, epochs=LOCAL_EPOCHS, verbose=0)
    train_time = time.time() - t0

    updated_weights = model.get_weights()
    _save_weights(out_weights_path, updated_weights)
    comm_bytes = int(sum(w.nbytes for w in updated_weights))  # حجم ما يُرسَل للخادم

    with open(out_info_path, "w") as f:
        json.dump({
            "num_examples": int(len(paths)),
            "train_time_sec": train_time,
            "comm_bytes_upload": comm_bytes,
            "ram_gb": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3),
        }, f)


def _eval_target(weights_path, out_info_path, round_num):
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    weights = _load_weights(weights_path)
    model = _build_model()
    model.set_weights(weights)

    test_ds = _make_dataset(X_test, y_test, BATCH_SIZE)

    t0 = time.time()
    loss, acc = model.evaluate(test_ds, verbose=0)
    eval_time = time.time() - t0

    # زمن استدلال دفعة واحدة فقط — يقارب زمن استجابة تطبيق هاتف مزارع لصورة واحدة
    one_batch = next(iter(test_ds.take(1)))
    xb, _ = one_batch
    t1 = time.time()
    _ = model.predict(xb, verbose=0)
    infer_time_per_batch = time.time() - t1
    infer_time_per_image = infer_time_per_batch / xb.shape[0]

    print(f"[الجولة {round_num}] دقة النموذج الاتحادي = {acc:.4f} | "
          f"زمن الاستدلال لكل صورة = {infer_time_per_image*1000:.2f} ms")

    with open(out_info_path, "w") as f:
        json.dump({
            "loss": float(loss),
            "accuracy": float(acc),
            "eval_time_sec": eval_time,
            "infer_time_per_image_ms": infer_time_per_image * 1000,
            "ram_gb": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3),
        }, f)


# -------------------- أدوات تشغيل عملية فرعية معزولة والانتظار حتى تنتهي --------------------
def _error_capturing_wrapper(target, error_path, args):
    """يُنفَّذ هذا الغلاف داخل العملية الفرعية: يشغّل الهدف الحقيقي، ويكتب أي خطأ لملف نصي."""
    try:
        target(*args)
    except Exception:
        import traceback
        with open(error_path, "w") as f:
            f.write(traceback.format_exc())
        raise


def run_isolated(target, args):
    ctx = mp.get_context("spawn")   # spawn إجباري: يضمن عملية جديدة تمامًا (عكس fork)
    error_path = os.path.join(CKPT_DIR, "_last_error.txt")
    if os.path.exists(error_path):
        os.remove(error_path)

    p = ctx.Process(target=_error_capturing_wrapper, args=(target, error_path, args))
    p.start()
    p.join()
    if p.exitcode != 0:
        detail = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                detail = "\n\nالخطأ الحقيقي داخل العملية الفرعية:\n" + f.read()
        raise RuntimeError(
            f"العملية الفرعية {target.__name__} فشلت بكود خروج {p.exitcode}{detail}"
        )


def average_weights(list_of_weights, list_of_sizes):
    total = sum(list_of_sizes)
    avg = []
    for layer_weights in zip(*list_of_weights):
        weighted_sum = sum(w * (n / total) for w, n in zip(layer_weights, list_of_sizes))
        avg.append(weighted_sum)
    return avg


# -------------------- الحلقة الرئيسية --------------------
if __name__ == "__main__":
    global_weights_path = os.path.join(CKPT_DIR, "global_weights.npz")
    eval_info_path = os.path.join(CKPT_DIR, "eval_info.json")

    results_rows = []
    start_round = 1

    # -------------------- فحص وجود تجربة سابقة للاستئناف منها --------------------
    if os.path.exists(RESULTS_CSV) and os.path.exists(global_weights_path):
        import pandas as pd
        prev_df = pd.read_csv(RESULTS_CSV)
        results_rows = prev_df.to_dict("records")
        last_round = int(prev_df["round"].max())
        start_round = last_round + 1
        print(f"✅ تم العثور على تجربة سابقة منتهية عند الجولة {last_round}.")
        print(f"   سيتم الاستئناف تلقائيًا من الجولة {start_round} حتى الهدف {TARGET_ROUNDS}، "
              f"باستخدام الأوزان المحفوظة (لن تُفقد أي جولة سابقة).")
    else:
        print("لا توجد تجربة سابقة محفوظة — بدء المحاكاة من الصفر.")
        print(f"بدء المحاكاة (عمليات معزولة): {NUM_CLIENTS} عميل، هدف {TARGET_ROUNDS} جولة، "
              f"{LOCAL_EPOCHS} حقبة محلية لكل جولة")

        run_isolated(_init_weights_target, (global_weights_path,))
        run_isolated(_eval_target, (global_weights_path, eval_info_path, 0))
        with open(eval_info_path) as f:
            eval_info = json.load(f)
        results_rows.append({
            "round": 0, "accuracy": eval_info["accuracy"], "loss": eval_info["loss"],
            "eval_time_sec": eval_info["eval_time_sec"],
            "infer_time_per_image_ms": eval_info["infer_time_per_image_ms"],
            "total_train_time_sec": 0.0, "total_comm_bytes": 0,
            "parent_ram_gb": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3),
        })
        import pandas as pd
        pd.DataFrame(results_rows).to_csv(RESULTS_CSV, index=False)

    if start_round > TARGET_ROUNDS:
        print(f"\nالتجربة مكتملة بالفعل حتى الجولة {TARGET_ROUNDS}. لا حاجة لأي تدريب إضافي. "
              f"ارفع TARGET_ROUNDS لعدد أكبر إن أردت الاستمرار أبعد من ذلك.")
    else:
        for round_num in range(start_round, TARGET_ROUNDS + 1):
            print(f"\n--- بدء الجولة {round_num} ---")
            client_weight_paths, client_sizes = [], []
            total_train_time, total_comm_bytes = 0.0, 0

            for cid in range(NUM_CLIENTS):
                out_w_path = os.path.join(CKPT_DIR, f"client_{cid}_weights.npz")
                out_info_path = os.path.join(CKPT_DIR, f"client_{cid}_info.json")

                run_isolated(_client_target, (cid, global_weights_path, out_w_path, out_info_path))

                with open(out_info_path) as f:
                    info = json.load(f)

                client_weight_paths.append(out_w_path)
                client_sizes.append(info["num_examples"])
                total_train_time += info["train_time_sec"]
                total_comm_bytes += info["comm_bytes_upload"]
                print(f"  اكتمل العميل {cid} ({info['num_examples']} صورة) | "
                      f"زمن تدريبه: {info['train_time_sec']:.1f}s | "
                      f"ذاكرة عمليته الفرعية: {info['ram_gb']:.2f} GB")

            # التجميع (FedAvg) يحدث في العملية الأب بـ numpy فقط — بدون TensorFlow
            client_weights = [_load_weights(p) for p in client_weight_paths]
            new_global_weights = average_weights(client_weights, client_sizes)
            _save_weights(global_weights_path, new_global_weights)
            del client_weights, new_global_weights
            gc.collect()

            run_isolated(_eval_target, (global_weights_path, eval_info_path, round_num))
            with open(eval_info_path) as f:
                eval_info = json.load(f)

            # حفظ النموذج الكامل القابل للاستخدام المباشر — في عملية معزولة أيضًا
            latest_model_path = os.path.join(MODELS_DIR, "latest_global_model.keras")
            run_isolated(_save_full_keras_model_target, (global_weights_path, latest_model_path))
            if round_num % SNAPSHOT_EVERY == 0 or round_num == TARGET_ROUNDS:
                snapshot_path = os.path.join(MODELS_DIR, f"global_model_round_{round_num}.keras")
                run_isolated(_save_full_keras_model_target, (global_weights_path, snapshot_path))
                print(f"  💾 نسخة نموذج كاملة محفوظة: {snapshot_path}")

            parent_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            print(f"  ذاكرة العملية الأب (يجب أن تبقى مستقرة): {parent_ram:.2f} GB")

            results_rows.append({
                "round": round_num, "accuracy": eval_info["accuracy"], "loss": eval_info["loss"],
                "eval_time_sec": eval_info["eval_time_sec"],
                "infer_time_per_image_ms": eval_info["infer_time_per_image_ms"],
                "total_train_time_sec": total_train_time, "total_comm_bytes": total_comm_bytes,
                "parent_ram_gb": parent_ram,
            })

            # حفظ تراكمي بعد كل جولة — لا تفقد أي نتيجة حتى لو انقطعت الجلسة لاحقًا
            import pandas as pd
            pd.DataFrame(results_rows).to_csv(RESULTS_CSV, index=False)

    print("\nانتهت المحاكاة الاتحادية.")
    print(f"سجل كامل مقاييس كل جولة محفوظ في: {RESULTS_CSV}")