Here's a complete breakdown:                                                                                                                                                         
                                                                                                                                                                                                                    
  ---                                                                                                                                                                                                               
  Failures: 8 total, 3 distinct root causes — all Keras 3 API breaks                                                                                                                                                  
  The project was written for Keras 2, but tensorflow==2.20.0 ships Keras 3 as tensorflow.keras. All three failures are consequences of that mismatch.                                                              
                                                                                                                                                                                                                    
  ---
  1. AveragePooling2D() — pool_size is now required (4 ERRORs)

  Affected tests: test_loading_parsed_model, test_parsing, test_normalizing, test_inisim

  conftest.py:166 and :204 call AveragePooling2D() with no arguments. In Keras 2 the pool_size defaulted to (2, 2). In Keras 3 it's a required positional arg:

  TypeError: AveragePooling2D.__init__() missing 1 required positional argument: 'pool_size'

  ---
  2. dtype='string' in model.fit() with a string loss name (3 ERRORs)

  Affected tests: test_conv1d, test_maxpool, test_maxpool_fallback

  The compile calls in test_conv.py:49 and test_pooling.py use:
  model.compile('adam', 'categorical_crossentropy', ['accuracy'])
  In Keras 3.14 + TF 2.20, this is a bug: when the loss is passed as a string and no loss_weights are given, the loss string key ends up being passed as the loss_weight tensor in compile_utils.py. During the     
  training step, Keras calls ops.multiply(loss_value, loss_weight) and then tries to promote dtypes ('float32', 'string'), which fails:

  ValueError: dtype='string' is not a valid dtype for Keras type promotion.

  ---
  3. DirectoryIterator.next() removed (1 FAILURE)

  Affected test: test_get_dataset_from_png

  snntoolbox/datasets/utils.py:125 calls the Python-2-style .next() method:
  ImageDataGenerator(rescale=rs).flow_from_directory(**dataflow_kwargs).next()[0]
  Keras 3 dropped .next() from iterators. It raises:
  AttributeError: 'DirectoryIterator' object has no attribute 'next'
  The fix is the built-in next(...).

  ---
  Summary table

  ┌─────┬─────────────────────────────────────────┬────────────────────────────────────┬────────────────┐
  │  #  │                  Error                  │             Root cause             │ Affected count │
  ├─────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ 1   │ TypeError: AveragePooling2D...pool_size │ Keras 3 removed default            │ 4 ERRORs       │
  ├─────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ 2   │ ValueError: dtype='string'              │ Keras 3.14 bug w/ string loss name │ 3 ERRORs       │
  ├─────┼─────────────────────────────────────────┼────────────────────────────────────┼────────────────┤
  │ 3   │ AttributeError: ...no attribute 'next'  │ Keras 3 removed .next()            │ 1 FAILED       │
  └─────┴─────────────────────────────────────────┴────────────────────────────────────┴────────────────┘

  The pyproject.toml currently pins tensorflow>=2.20.0 which forces Keras 3. You have two paths: pin back to tensorflow~=2.15 (last release that bundled Keras 2), or patch the three spots above to be Keras 3     
  compatible. Want me to do either?
