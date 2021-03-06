?	%?????@%?????@!%?????@	???y??????y???!???y???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6%?????@:?Fve?w@1?G?z??b@A@1?d????I?3??Xv(@YQMI?????*	Yd;?/??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator<jL??%@!??q??X@)<jL??%@1??q??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchx?Go????!?³*????)x?Go????1?³*????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???Y???!???=??)eo)狝?1[?M?QE??:Preprocessing2F
Iterator::Model{??v? ??!?>??l7??)ҩ+??y??1??/??B??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?!r?z%@!?L"?X@)ݚt["l?1???lk??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???y???I????Z#R@Q? ?g?d;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:?Fve?w@:?Fve?w@!:?Fve?w@      ??!       "	?G?z??b@?G?z??b@!?G?z??b@*      ??!       2	@1?d????@1?d????!@1?d????:	?3??Xv(@?3??Xv(@!?3??Xv(@B      ??!       J	QMI?????QMI?????!QMI?????R      ??!       Z	QMI?????QMI?????!QMI?????b      ??!       JGPUY???y???b q????Z#R@y? ?g?d;@?"7
model_8/conv2d_429/Conv2DConv2D?5xx?ب?!?5xx?ب?0"l
Bgradient_tape/model_8/batch_normalization_429/FusedBatchNormGradV3FusedBatchNormGradV3:????m??!?Sz??ǰ?"7
model_8/conv2d_429/BiasAddBiasAdd????XW??!??T̡???"h
<gradient_tape/model_8/conv2d_433/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ޟ3O??!q??r?\??0"f
;gradient_tape/model_8/conv2d_433/Conv2D/Conv2DBackpropInputConv2DBackpropInput?lun???!ZlA???0"V
0model_8/batch_normalization_429/FusedBatchNormV3FusedBatchNormV3?](?V??!??o?ȷ??"f
;gradient_tape/model_8/conv2d_431/Conv2D/Conv2DBackpropInputConv2DBackpropInput?!???*??!?)??']??0"7
model_8/conv2d_438/Conv2DConv2Ds{J?%??!?<?S????0"7
model_8/conv2d_441/Conv2DConv2D?֕S? ??!??8????0"7
model_8/conv2d_431/Conv2DConv2D??????!	?h??$??0Q      Y@Y??X????a\???`?X@q_??F??C@y?U$?n?"?

both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?39.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 