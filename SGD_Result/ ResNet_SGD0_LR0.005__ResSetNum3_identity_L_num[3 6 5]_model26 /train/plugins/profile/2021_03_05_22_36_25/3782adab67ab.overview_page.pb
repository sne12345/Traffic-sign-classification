?	|&??)|?@|&??)|?@!|&??)|?@	?︲D???︲D??!?︲D??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6|&??)|?@?+J	)y@1˅ʿ?a@AZ??/-???I?ZdK)@YT? Pō??*	?K7??'?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratoruX??@!Z?yn?bX@)uX??@1Z?yn?bX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?i??_=??!?????)??)?i??_=??1?????)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?4`??i??!?MuQp?@)?-?R???1?N{?5???:Preprocessing2F
Iterator::Model?Ēr?9??!G????E@)??QF\ z?1??ŋ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?@-#@!?????eX@){?\?&?k?1<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 71.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?︲D??I[??mْR@Q????w?9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?+J	)y@?+J	)y@!?+J	)y@      ??!       "	˅ʿ?a@˅ʿ?a@!˅ʿ?a@*      ??!       2	Z??/-???Z??/-???!Z??/-???:	?ZdK)@?ZdK)@!?ZdK)@B      ??!       J	T? Pō??T? Pō??!T? Pō??R      ??!       Z	T? Pō??T? Pō??!T? Pō??b      ??!       JGPUY?︲D??b q[??mْR@y????w?9@?"9
model_30/conv2d_1612/Conv2DConv2DQ???맩?!Q???맩?0"n
Dgradient_tape/model_30/batch_normalization_1612/FusedBatchNormGradV3FusedBatchNormGradV3??C?|???!V??S??"9
model_30/conv2d_1612/BiasAddBiasAdd??tZ?0??!???Z19??"j
>gradient_tape/model_30/conv2d_1616/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter猨M????!O엄??0"h
=gradient_tape/model_30/conv2d_1616/Conv2D/Conv2DBackpropInputConv2DBackpropInput ??+I#??!?=?xܹ?0"h
=gradient_tape/model_30/conv2d_1624/Conv2D/Conv2DBackpropInputConv2DBackpropInputF?f?`??!X{?$???0"9
model_30/conv2d_1624/Conv2DConv2D??n???!?=6?]??0"9
model_30/conv2d_1614/Conv2DConv2D%U?y???!???5B??0"X
2model_30/batch_normalization_1612/FusedBatchNormV3FusedBatchNormV3I???߅?!@??c@k??"9
model_30/conv2d_1621/Conv2DConv2DKl??Յ?!?ğ???0Q      Y@Y?va?????a&z6 ??X@q?????G@y?Ġ???q?"?

both?Your program is POTENTIALLY input-bound because 71.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?46.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 