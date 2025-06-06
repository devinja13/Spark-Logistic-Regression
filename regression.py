import re
import numpy as np
import heapq

# ——— Load and Parse Corpus ————————————————————————————————————————
# Load all 19,997 documents in the corpus
corpus = sc.textFile("s3://luisguzmannateras/Assignment5/TestingDataOneLinePerDoc.txt")
# corpus = sc.textFile("s3://luisguzmannateras/Assignment5/SmallTrainingDataOneLinePerDoc.txt")

# Filter lines containing ‘id’ and extract (docID, raw_text)
valid_lines = corpus.filter(lambda line: 'id' in line)
key_and_text = valid_lines.map(lambda line: (
    line[line.index('id="') + 4 : line.index('" url=')],
    line[line.index('">') + 2:]
))

# Split each text into a list of lowercase words (keep only A–Z)
NON_ALPHA = re.compile(r'[^a-zA-Z]')
key_and_list_of_words = key_and_text.map(
    lambda kv: (str(kv[0]), NON_ALPHA.sub(' ', kv[1]).lower().split())
)

# ——— Build Top-20,000 Word Dictionary ———————————————————————————————
# Flatten to (word, 1) pairs
all_words = key_and_list_of_words.flatMap(lambda kv: ((w, 1) for w in kv[1]))

# Count total occurrences per word: (word, count)
all_counts = all_words.reduceByKey(lambda a, b: a + b)

# Take top 20,000 words, ordered by descending count, then lexicographically
top_words = all_counts.takeOrdered(20000, key=lambda wc: (-wc[1], wc[0]))

# Create an RDD of indices 0…19999 and map to (word, index)
twenty_k = sc.parallelize(range(20000))
dictionary = twenty_k.map(lambda i: (top_words[i][0], i))

# Collect dictionary to a local map for quick lookups/debugging
dict_map = dictionary.collectAsMap()
print("proceeding:", dict_map["proceeding"])
print("orders:", dict_map["orders"])
print("formula:", dict_map["formula"])
print("tribunal:", dict_map["tribunal"])

# Build Python dictionaries for word means & indices
word_mean = {word: count / 19997 for word, count in top_words}
word_index = {word: i for i, (word, _) in enumerate(top_words)}

num_docs = key_and_list_of_words.count()
word_mean_array = np.zeros(20000, dtype=float)
for word, idx in word_index.items():
    word_mean_array[idx] = word_mean[word]

# ——— Create Sparse Count Vectors per Document —————————————————————————
def create_vector(doc_kv, word_index):
    """
    Given (docID, [words]), return (docID, count_vector) of length 20000.
    """
    doc_id, words = doc_kv
    vec = np.zeros(20000, dtype=int)
    for w in words:
        idx = word_index.get(w)
        if idx is not None:
            vec[idx] += 1
    return doc_id, vec

# doc_vect: RDD[(docID, count_vector)]
doc_vect = key_and_list_of_words.map(lambda kv: create_vector(kv, word_index))

# Compute per-word variance components to get standard deviation
square_sum = doc_vect.map(lambda kv: (kv[0], (kv[1] - word_mean_array) ** 2))
square_sum_vec = square_sum.map(lambda kv: kv[1]).reduce(lambda a, b: a + b)
stdev = np.sqrt(square_sum_vec / (num_docs - 1))
print("Zero variances:", [v for v in stdev if v == 0])

# ——— TF-IDF Computation ——————————————————————————————————————————
# Total words per doc
total_words = doc_vect.map(lambda kv: (kv[0], kv[1].sum()))
total_words_doc_vect = doc_vect.join(total_words)  # (docID, (count_vec, total_count))

# TF: (docID, tf_scalar_array) where tf_scalar_array = count_vec / total_count
tf = total_words_doc_vect.map(lambda kv: (kv[0], kv[1][0] / kv[1][1]))

# IDF: count how many docs contain each word
unique_words = key_and_list_of_words.map(lambda kv: (kv[0], set(kv[1])))
key_words_values = unique_words.flatMap(lambda kv: ((w, 1) for w in kv[1]))
words_to_doc = key_words_values.reduceByKey(lambda a, b: a + b)  # (word, doc_freq)

total_docs = doc_vect.count()
idf_rdd = words_to_doc.map(lambda kv: (kv[0], np.log(total_docs / kv[1])))
idf_dict = idf_rdd.collectAsMap()

# Build IDF vector aligned with word_index order
idf_vector = np.array([
    idf_dict.get(word, 0.0)
    for word in sorted(word_index, key=word_index.get)
])
idf_vector_broadcast = sc.broadcast(idf_vector)

# TF-IDF: (docID, tfidf_array) = tf_array * idf_vector
tfidf = tf.map(lambda kv: (kv[0], kv[1] * idf_vector_broadcast.value))

# Normalize TF-IDF and append 1.0 bias term
sums = tfidf.map(lambda kv: kv[1]).reduce(lambda a, b: a + b)
num_docs = tfidf.count()
mean_vector = sums / num_docs

sum_squared_diff = tfidf.map(lambda kv: (kv[1] - mean_vector) ** 2).reduce(lambda a, b: a + b)
variance_vector = sum_squared_diff / (num_docs - 1)
stdev_vector = np.sqrt(variance_vector)

normalized_data_tfidf = tfidf.map(lambda kv: (
    kv[0],
    np.append(
        np.where(stdev_vector != 0, (kv[1] - mean_vector) / stdev_vector, 0.0),
        1.0  # bias
    )
))

# ——— Logistic Regression Training ————————————————————————————————————
y_regex = re.compile(r'^AU[0-9]+')


def compute_gradient(r, data_rdd, lam):
    """
    Compute gradient for L2-regularized logistic regression.
    r: parameter vector (length 20001)
    data_rdd: RDD[(docID, feature_vector_with_bias)]
    lam: regularization coefficient
    """
    def per_doc_grad(kv):
        doc_id, vec = kv
        y_val = 1 if y_regex.match(doc_id) else 0
        z = np.dot(vec, r)
        p = 1 / (1 + np.exp(-z))
        return (vec * (p - y_val)) / num_docs

    grad_rdd = data_rdd.map(per_doc_grad)
    sum_grad = grad_rdd.reduce(lambda a, b: a + b)
    reg_term = lam * r.copy()
    reg_term[-1] = 0.0  # do not regularize bias
    return sum_grad + reg_term


def compute_log_likelihood(r, data_rdd):
    """
    Compute total log-likelihood over data given parameter r.
    """
    def point_llh(kv):
        doc_id, vec = kv
        y_val = 1 if y_regex.match(doc_id) else 0
        z = np.dot(vec, r)
        s = 1 / (1 + np.exp(-z))
        s = np.clip(s, 1e-15, 1 - 1e-15)
        return y_val * np.log(s) + (1 - y_val) * np.log(1 - s)

    return data_rdd.map(point_llh).sum()


# Initialize parameters
r = np.zeros(20001)
alpha = 0.01
i = 0
new_log = 0
delta = 100

print("Initial gradient norm:",
      np.linalg.norm(compute_gradient(r, normalized_data_tfidf, 0.001)))

# Gradient-descent loop
while delta > 1e-20 and i < 1000:
    grad = compute_gradient(r, normalized_data_tfidf, 0.1)
    old_r = r.copy()
    r = old_r - alpha * grad
    delta = np.linalg.norm(r - old_r)

    old_log = new_log
    new_log = -compute_log_likelihood(r, normalized_data_tfidf)
    if old_log < new_log:
        alpha *= 1.05
    else:
        alpha *= 0.5

    print(f"[Iteration {i}] Δ = {delta:.3e}, LogLik = {new_log:.3e}, α = {alpha:.2e}")
    i += 1

# Extract top 20 coefficients (exclude bias at index 20000)
coef = r[:-1]
top20_idx = heapq.nlargest(20, range(len(coef)), key=lambda idx: coef[idx])
index_to_word = {idx: w for w, idx in word_index.items()}
top_words = [(index_to_word[idx], coef[idx]) for idx in top20_idx]
print("Most impactful words:", top_words)

# ——— Evaluation on Test Set ————————————————————————————————————————
testing = sc.textFile("s3://luisguzmannateras/Assignment5/SmallTrainingDataOneLinePerDoc.txt")
test_valid_lines = testing.filter(lambda line: 'id' in line)
test_key_and_text = test_valid_lines.map(lambda line: (
    line[line.index('id="') + 4 : line.index('" url=')],
    line[line.index('">') + 2:]
))

test_key_and_list_of_words = test_key_and_text.map(lambda kv: (
    str(kv[0]),
    NON_ALPHA.sub(' ', kv[1]).lower().split()
))

test_doc_vect = test_key_and_list_of_words.map(lambda kv: create_vector(kv, word_index))
test_total_words = test_doc_vect.map(lambda kv: (kv[0], kv[1].sum()))
test_total_words_doc_vect = test_doc_vect.join(test_total_words)
test_tf = test_total_words_doc_vect.map(lambda kv: (kv[0], kv[1][0] / kv[1][1]))

test_tfidf = test_tf.map(lambda kv: (kv[0], kv[1] * idf_vector_broadcast.value))
test_normalized = test_tfidf.map(lambda kv: (
    kv[0],
    np.append(
        np.where(stdev_vector != 0, (kv[1] - mean_vector) / stdev_vector, 0.0),
        1.0
    )
))

num_matches = test_normalized.filter(lambda kv: y_regex.match(kv[0])).count()
print("Number of matching documents:", num_matches)

res = test_normalized.map(lambda kv: (
    kv[0],
    1 if np.dot(kv[1], r) > 0.25 else 0
))
res = res.map(lambda kv: (kv[0], ((1 if y_regex.match(kv[0]) else 0), kv[1])))

counts = res.map(lambda kv: kv[1]).countByValue()
TP = counts.get((1, 1), 0)
FP = counts.get((0, 1), 0)
FN = counts.get((1, 0), 0)
print("TP:", TP, "FP:", FP, "FN:", FN)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
print("F1 Score:", f1)

false_positives = res.filter(lambda kv: kv[1][0] == 0 and kv[1][1] == 1)
false_positive_text = false_positives.join(test_key_and_text)
for doc_id, ((true_label, pred_label), text) in false_positive_text.take(3):
    print("---- FALSE POSITIVE ----")
    print("DocID:", doc_id)
    print("Text:", text)
