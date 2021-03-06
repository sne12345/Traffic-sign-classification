?	#??o{@#??o{@!#??o{@	?c?G]???c?G]??!?c?G]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6#??o{@ᶶ?<?s@1?E?  \@A?A??v???I??:M?$@Yw?*2: ??*	?C?lǯ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator@1?d?@!7Z??X@)@1?d?@17Z??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Rxа?!b??u?T??)???Rxа?1b??u?T??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?_?5?!??!Vi?????)??m??E??1???}???:Preprocessing2F
Iterator::Model?t ??շ?!}??V???)nR?X?;{?1m????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?-????@!???λ?X@)an?r?l?1??Vz???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 72.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?c?G]??I?Ő?p?R@Q4`???9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ᶶ?<?s@ᶶ?<?s@!ᶶ?<?s@      ??!       "	?E?  \@?E?  \@!?E?  \@*      ??!       2	?A??v????A??v???!?A??v???:	??:M?$@??:M?$@!??:M?$@B      ??!       J	w?*2: ??w?*2: ??!w?*2: ??R      ??!       Z	w?*2: ??w?*2: ??!w?*2: ??b      ??!       JGPUY?c?G]??b q?Ő?p?R@y4`???9@?"8
model_10/conv2d_545/Conv2DConv2D?҉*n??!?҉*n??0"m
Cgradient_tape/model_10/batch_normalization_545/FusedBatchNormGradV3FusedBatchNormGradV3??]????!??i0?2??"8
model_10/conv2d_545/BiasAddBiasAdd??H ???!?p?9,???"i
=gradient_tape/model_10/conv2d_549/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?+??w??!??m?"???0"g
<gradient_tape/model_10/conv2d_549/Conv2D/Conv2DBackpropInputConv2DBackpropInput?"?a??!??(1???0"W
1model_10/batch_normalization_545/FusedBatchNormV3FusedBatchNormV3?ǻ???!&??S??"g
<gradient_tape/model_10/conv2d_547/Conv2D/Conv2DBackpropInputConv2DBackpropInputb?
g6??!L?? b??0"8
model_10/conv2d_547/Conv2DConv2D?ī`???!?<??S???0"8
model_10/conv2d_551/Conv2DConv2D?ɦ?????!5?zA@???0"8
model_10/conv2d_554/Conv2DConv2D?P?0????!B???O??0Q      Y@Yk?4w?_??a?,#???X@q=?٥?H@ye??u'xv?"?

both?Your program is POTENTIALLY input-bound because 72.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?49.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 