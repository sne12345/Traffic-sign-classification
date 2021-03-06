?	en????@en????@!en????@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-en????@??a?x@1??5Φ?l@AQ???????I?nK??@*	e;?OM??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator͔???@!jHy?X@)͔???@1jHy?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????????!Xt???)????????1Xt???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!Z??????)p@KW????1Nlg1?v??:Preprocessing2F
Iterator::Model
dv?S??!;???3??)	^?z?1MB?0c??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? ???@!Ǹ?50?X@)?4a??o?1????o???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIv?=?O@Q?????@B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??a?x@??a?x@!??a?x@      ??!       "	??5Φ?l@??5Φ?l@!??5Φ?l@*      ??!       2	Q???????Q???????!Q???????:	?nK??@?nK??@!?nK??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qv?=?O@y?????@B@?"8
model_17/conv2d_909/Conv2DConv2D?M{?u??!?M{?u??0"i
=gradient_tape/model_17/conv2d_921/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!1?S~???![?s?0??0"i
=gradient_tape/model_17/conv2d_915/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP?T?c???!???V????0"i
=gradient_tape/model_17/conv2d_918/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>6?~????!R?w6Y\??0"i
=gradient_tape/model_17/conv2d_911/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter÷??????!C '???0"g
<gradient_tape/model_17/conv2d_914/Conv2D/Conv2DBackpropInputConv2DBackpropInput'??E?Z??!??G9B??0"g
<gradient_tape/model_17/conv2d_920/Conv2D/Conv2DBackpropInputConv2DBackpropInputtm???M??!?B?????0"g
<gradient_tape/model_17/conv2d_917/Conv2D/Conv2DBackpropInputConv2DBackpropInput?ߌ^2H??!??֖+??0"i
=gradient_tape/model_17/conv2d_917/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ӆ ???!KI2??u??0"i
=gradient_tape/model_17/conv2d_920/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterE?B=?x??!?xCb???0Q      Y@Y?]?/7???a??A#??X@qn?b?xkL@y[D?j?b?"?

both?Your program is POTENTIALLY input-bound because 62.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?56.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 