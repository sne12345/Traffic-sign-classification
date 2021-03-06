?	?f???{@?f???{@!?f???{@	? ??56??? ??56??!? ??56??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?f???{@?W???s@1???!Ɩ^@A???|@???I@??>?&@Y:???u??*	????ͺ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator? d??!@!?E#h?X@)? d??!@1?E#h?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???C?r??!	????)???C?r??1	????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn??Sr??!]A+????)???~???1avf=???:Preprocessing2F
Iterator::Model%?/????!ƶ??U??)O=?බ??1?VgM???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? ?S??!@!%?騮X@)???O??m?1?}????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9? ??56??If?A??"R@QH??Q[b;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?W???s@?W???s@!?W???s@      ??!       "	???!Ɩ^@???!Ɩ^@!???!Ɩ^@*      ??!       2	???|@??????|@???!???|@???:	@??>?&@@??>?&@!@??>?&@B      ??!       J	:???u??:???u??!:???u??R      ??!       Z	:???u??:???u??!:???u??b      ??!       JGPUY? ??56??b qf?A??"R@yH??Q[b;@?"8
model_12/conv2d_645/Conv2DConv2Dv?G?p??!v?G?p??0"m
Cgradient_tape/model_12/batch_normalization_645/FusedBatchNormGradV3FusedBatchNormGradV3!A???4??!??¿?=??"8
model_12/conv2d_645/BiasAddBiasAdd?+ |????!2?B/u???"i
=gradient_tape/model_12/conv2d_649/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}?S?}??!???Y(???0"g
<gradient_tape/model_12/conv2d_649/Conv2D/Conv2DBackpropInputConv2DBackpropInput`<?k:??!?s?s????0"W
1model_12/batch_normalization_645/FusedBatchNormV3FusedBatchNormV3NG?}//??!3?{k?X??"8
model_12/conv2d_651/Conv2DConv2D?9???!?h??4??0"8
model_12/conv2d_654/Conv2DConv2Dx?yQ???!??????0"8
model_12/conv2d_647/Conv2DConv2D^?O???!p LQ????0"g
<gradient_tape/model_12/conv2d_651/Conv2D/Conv2DBackpropInputConv2DBackpropInput??ߝ???!}1K/?Z??0Q      Y@Yk?4w?_??a?,#???X@q0ƛWC@yvwL??ut?"?

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
Refer to the TF2 Profiler FAQb?38.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 