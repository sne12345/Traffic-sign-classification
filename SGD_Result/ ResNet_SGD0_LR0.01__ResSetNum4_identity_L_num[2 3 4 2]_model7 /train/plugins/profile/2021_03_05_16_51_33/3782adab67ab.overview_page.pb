?	??fdPC|@??fdPC|@!??fdPC|@	?.??:???.??:??!?.??:??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??fdPC|@o????s@1?]~?_@A<?y?9[??I??tw?E%@Y-?2????*	h??|???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?h8en?%@!??????X@)?h8en?%@1??????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?^?????!?=?????)?^?????1?=?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism6??,
???!p?f?7??)?6ɏ???1A??1]???:Preprocessing2F
Iterator::Modelӣ??????!???-S??)?y?'Lx?1M?^?xn??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?W?L?%@!?|H???X@)?%8???m?1 *@??ߠ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?.??:??Iy??J4R@Q맨T??;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o????s@o????s@!o????s@      ??!       "	?]~?_@?]~?_@!?]~?_@*      ??!       2	<?y?9[??<?y?9[??!<?y?9[??:	??tw?E%@??tw?E%@!??tw?E%@B      ??!       J	-?2????-?2????!-?2????R      ??!       Z	-?2????-?2????!-?2????b      ??!       JGPUY?.??:??b qy??J4R@y맨T??;@?"8
model_11/conv2d_595/Conv2DConv2D\?(onz??!\?(onz??0"m
Cgradient_tape/model_11/batch_normalization_595/FusedBatchNormGradV3FusedBatchNormGradV3???Zt7??!?vɅK??"8
model_11/conv2d_595/BiasAddBiasAdd	q3?S???!????U???"i
=gradient_tape/model_11/conv2d_599/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter<O?]???!?n??As??0"g
<gradient_tape/model_11/conv2d_599/Conv2D/Conv2DBackpropInputConv2DBackpropInput6_??N??!X?Ҍ~??0"8
model_11/conv2d_601/Conv2DConv2DJ?G??#??!?6t??@??0"8
model_11/conv2d_597/Conv2DConv2D??????!GX޺???0"W
1model_11/batch_normalization_595/FusedBatchNormV3FusedBatchNormV3? "???!d?????"8
model_11/conv2d_604/Conv2DConv2DX?M?.???!nAd???0"g
<gradient_tape/model_11/conv2d_601/Conv2D/Conv2DBackpropInputConv2DBackpropInput??i^???!???ܕA??0Q      Y@Yk?4w?_??a?,#???X@q?E?vUA@y??B??<t?"?

both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?34.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 