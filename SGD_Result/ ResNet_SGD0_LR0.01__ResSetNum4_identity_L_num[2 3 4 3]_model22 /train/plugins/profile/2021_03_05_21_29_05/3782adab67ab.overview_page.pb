?	???W??@???W??@!???W??@	?Q	l?`???Q	l?`??!?Q	l?`??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???W??@?u?;?3v@1.??Ľj@A?8?ߡ(??Ie??? @Y?ʅʿ???*	? ?r???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?5??x&@!?x9?y?X@)?5??x&@1?x9?y?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchѯ??????!Y???????)ѯ??????1Y???????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???Im??!s
B}?g??)ܺ??:???1f?5?Ϯ??:Preprocessing2F
Iterator::Model??fF???!?GO;I??)a?$??z?1w???{??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???u6&@!pa??m?X@)_?Q?k?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Q	l?`??I?r??O@Q؊?30B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?u?;?3v@?u?;?3v@!?u?;?3v@      ??!       "	.??Ľj@.??Ľj@!.??Ľj@*      ??!       2	?8?ߡ(???8?ߡ(??!?8?ߡ(??:	e??? @e??? @!e??? @B      ??!       J	?ʅʿ????ʅʿ???!?ʅʿ???R      ??!       Z	?ʅʿ????ʅʿ???!?ʅʿ???b      ??!       JGPUY?Q	l?`??b q?r??O@y؊?30B@?"9
model_26/conv2d_1396/Conv2DConv2D?*???`??!?*???`??0"j
>gradient_tape/model_26/conv2d_1402/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????N??!?do??0"j
>gradient_tape/model_26/conv2d_1398/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?'???K??!?&]W
W??0"j
>gradient_tape/model_26/conv2d_1405/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQ??I??!??v???0"h
=gradient_tape/model_26/conv2d_1401/Conv2D/Conv2DBackpropInputConv2DBackpropInput?}?3??!????ܵ??0"h
=gradient_tape/model_26/conv2d_1404/Conv2D/Conv2DBackpropInputConv2DBackpropInputU?8#A??!-ʼ?????0"j
>gradient_tape/model_26/conv2d_1401/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?)0%~??!?d?4?>??0"j
>gradient_tape/model_26/conv2d_1404/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?<2qd??!|??G
???0"9
model_26/conv2d_1402/Conv2DConv2D@??????!`e?y???0"9
model_26/conv2d_1398/Conv2DConv2D	??????!?????$??0Q      Y@Y???????a?A?g?X@q???8?UA@y??օ>*g?"?

both?Your program is POTENTIALLY input-bound because 62.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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