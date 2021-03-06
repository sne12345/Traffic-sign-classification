?	??ފf?@??ފf?@!??ފf?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??ފf?@?-?8t@1????_h@A?R???Ư?Ir???!@*	j?t? ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator9`W??@!O???eX@)9`W??@1O???eX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchl_@/ܹ??!?????)l_@/ܹ??1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism-AF@?#??!?cX?g?@)?E??\??1?T%?t}??:Preprocessing2F
Iterator::Model?%?<??!?:xl??@)w?????}?1u???R??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapK?|% @!)>?\ciX@)??V?I?k?1??.?/Ү?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????s?O@QgC?UB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?-?8t@?-?8t@!?-?8t@      ??!       "	????_h@????_h@!????_h@*      ??!       2	?R???Ư??R???Ư?!?R???Ư?:	r???!@r???!@!r???!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????s?O@ygC?UB@?"9
model_23/conv2d_1243/Conv2DConv2D????.+??!????.+??0"j
>gradient_tape/model_23/conv2d_1249/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?'??????!ZlщT???0"j
>gradient_tape/model_23/conv2d_1245/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP?C7???!n1?Zb^??0"j
>gradient_tape/model_23/conv2d_1252/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'1ϹӁ??!??3I׾??0"h
=gradient_tape/model_23/conv2d_1251/Conv2D/Conv2DBackpropInputConv2DBackpropInput/??*7???!?,F????0"h
=gradient_tape/model_23/conv2d_1248/Conv2D/Conv2DBackpropInputConv2DBackpropInput???x???!N{N?????0"j
>gradient_tape/model_23/conv2d_1248/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?sz??Ҋ?!?"?%#]??0"j
>gradient_tape/model_23/conv2d_1251/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterĦ?[4͊?!?,?k?	??0"n
Dgradient_tape/model_23/batch_normalization_1243/FusedBatchNormGradV3FusedBatchNormGradV3??b#p???!?V?mͥ??"9
model_23/conv2d_1249/Conv2DConv2D9ԃs%???!???>??0Q      Y@Y6?NK?~??ag????X@q7?C6R?I@yC?'??bh?"?

both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?51.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 