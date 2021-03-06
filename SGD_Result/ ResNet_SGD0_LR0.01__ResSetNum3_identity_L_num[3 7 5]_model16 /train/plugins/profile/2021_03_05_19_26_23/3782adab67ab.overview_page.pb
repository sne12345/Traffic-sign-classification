?	?V*(??@?V*(??@!?V*(??@	????ꪚ?????ꪚ?!????ꪚ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?V*(??@?A?????@1&?ls?2o@A??q?߅??In4??@R@YʉvR~??*	????댹@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Z(???@!g?????X@)?Z(???@1g?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchDܜJ???!???La???)DܜJ???1???La???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismaQ??l??!`??&K??)t??q5???1LW h?v??:Preprocessing2F
Iterator::Model?3??k???!??ݹ??)?$>w??w?1?$?pZ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap̘?5?@!??ǈ?X@)??K?l?12T????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 71.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????ꪚ?I2??( ,R@Q?q???H;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?A?????@?A?????@!?A?????@      ??!       "	&?ls?2o@&?ls?2o@!&?ls?2o@*      ??!       2	??q?߅????q?߅??!??q?߅??:	n4??@R@n4??@R@!n4??@R@B      ??!       J	ʉvR~??ʉvR~??!ʉvR~??R      ??!       Z	ʉvR~??ʉvR~??!ʉvR~??b      ??!       JGPUY????ꪚ?b q2??( ,R@y?q???H;@?"9
model_20/conv2d_1077/Conv2DConv2DXm?T܋??!Xm?T܋??0"j
>gradient_tape/model_20/conv2d_1089/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}n?pԖ?!?u?n&0??0"j
>gradient_tape/model_20/conv2d_1086/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??b?AW??!?ۭ?㭲?0"j
>gradient_tape/model_20/conv2d_1083/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?8?????!?钣???0"j
>gradient_tape/model_20/conv2d_1079/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterV?u?w ??!?Qp?./??0"h
=gradient_tape/model_20/conv2d_1088/Conv2D/Conv2DBackpropInputConv2DBackpropInputy_?x???!? ????0"h
=gradient_tape/model_20/conv2d_1082/Conv2D/Conv2DBackpropInputConv2DBackpropInput'?l????!x??e?i??0"h
=gradient_tape/model_20/conv2d_1085/Conv2D/Conv2DBackpropInputConv2DBackpropInput"?Q9???!??SY????0"j
>gradient_tape/model_20/conv2d_1088/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterO30?9???!?v?c??0"9
model_20/conv2d_1086/Conv2DConv2DV9?Ĥփ?!?&?A?J??0Q      Y@Y|?t8Y2??a,?6?X@q:k?J??G@y??L9_?b?"?

both?Your program is POTENTIALLY input-bound because 71.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?47.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 