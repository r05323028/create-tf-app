# Create TF App Command Line Tools

This python package is used to create tensorflow side project.

## Installation

```
pip install create_tf_app
```

## Create side project 

After installing this package by `pip`, you can simply use `create-tf-app` cli to create a tensorflow app which uses the newest high-level api (i.e. `tf.data.Dataset` and `tf.estimator.Estimator`).

```
Usage: create-tf-app [OPTIONS]

Options:
  --app_name TEXT            The app name.  [required]
  --use_gpu BOOLEAN          This side project whether to use gpus.
  --use_example [iris]       The name of example project.
  --python_interpreter TEXT  Specify the python interpreter. (for VSCode)
  --help                     Show this message and exit.
```

## Example: Iris dataset

Use the cli to create example side project.

```
create-tf-app --app_name=iris_net --use_example=iris --python_interpreter=/your/venv/bin/python
```

In `train.py`, we used the `scikit-learn` package to load iris dataset.

```python
features, labels = load_iris(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=kwargs.get('test_size'))
train_data = (train_X, train_y)
eval_data = (test_X, test_y)
```

In `iris_net/__init__.py`, we already defined the `input_fn`, `forward`, `model_fn` which you need to design by yourself if you create an empty project.

`input_fn` function,
```python
if mode == tf.estimator.ModeKeys.TRAIN:
            
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(self.params['num_epochs']).batch(self.params['batch_size']).prefetch(self.params['batch_size'])

    return dataset
  

elif mode == tf.estimator.ModeKeys.EVAL:
  
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(self.params['num_epochs']).batch(self.params['batch_size']).prefetch(self.params['batch_size'])

    return dataset

...
```

`forward` function,
```python
with tf.name_scope('hidden_layers'):
    net = tf.layers.dense(features, 10, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=tf.nn.relu)

with tf.name_scope('output_layer'):
    net = tf.layers.dense(net, 3)

return net
```

`model_fn` function,
```python
if mode == tf.estimator.ModeKeys.TRAIN:
    
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, depth=3), logits)
    opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    

if mode == tf.estimator.ModeKeys.EVAL:
    
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, depth=3), logits)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            tf.one_hot(labels, depth=3), tf.one_hot(tf.argmax(logits, axis=1), depth=3), name='accuracy'),
        'precision': tf.metrics.precision(
            tf.one_hot(labels, depth=3), tf.one_hot(tf.argmax(logits, axis=1), depth=3), name='precision'),
        'recall': tf.metrics.recall(
            tf.one_hot(labels, depth=3), tf.one_hot(tf.argmax(logits, axis=1), depth=3), name='recall'),
        'f1_score': tf.contrib.metrics.f1_score(
            tf.one_hot(labels, depth=3), tf.one_hot(tf.argmax(logits, axis=1), depth=3), name='f1_score')
    }
    

if mode == tf.estimator.ModeKeys.PREDICT:
    
    predictions = tf.one_hot(tf.argmax(logits, axis=1), depth=3)
```

Finally, just use python script to train the model.

```
python train.py
```