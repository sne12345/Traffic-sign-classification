?	?$?9??@?$?9??@!?$?9??@	>^3?x??>^3?x??!>^3?x??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?$?9??@??25I?t@1	???g@A=?U?????ID??<??#@Yw?h?hs??*	????w?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????@?"@!0???X@)????@?"@10???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?M?#E??!?Oג`???)?M?#E??1?Oג`???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Z? m???!?,bǥ??)?4?($???1??R=????:Preprocessing2F
Iterator::Model?{???!???э??)2t??w?1?? x????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?]Pߺ"@!ۣl\??X@)G????i?1?8V????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9>^3?x??I??o?P@QzH?Y?A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??25I?t@??25I?t@!??25I?t@      ??!       "		???g@	???g@!	???g@*      ??!       2	=?U?????=?U?????!=?U?????:	D??<??#@D??<??#@!D??<??#@B      ??!       J	w?h?hs??w?h?hs??!w?h?hs??R      ??!       Z	w?h?hs??w?h?hs??!w?h?hs??b      ??!       JGPUY>^3?x??b q??o?P@yzH?Y?A@?"9
model_22/conv2d_1193/Conv2DConv2D'?X.V??!'?X.V??0"j
>gradient_tape/model_22/conv2d_1199/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterUD?????!?6?S)??0"j
>gradient_tape/model_22/conv2d_1202/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4>m)???!?!?,????0"j
>gradient_tape/model_22/conv2d_1195/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?~?뼛?!ѣe$ٓ??0"h
=gradient_tape/model_22/conv2d_1201/Conv2D/Conv2DBackpropInputConv2DBackpropInput,)?q???!{???3???0"h
=gradient_tape/model_22/conv2d_1198/Conv2D/Conv2DBackpropInputConv2DBackpropInputt%??ގ?!?%?L ???0"j
>gradient_tape/model_22/conv2d_1198/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterm?W?F??!??re?Z??0"j
>gradient_tape/model_22/conv2d_1201/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?
s??!?e!???0"9
model_22/conv2d_1195/Conv2DConv2D??Hg~???!F??|????0"9
model_22/conv2d_1199/Conv2DConv2D:??????!Z???`I??0Q      Y@Y6?NK?~??ag????X@q?B?\?;B@yk
?h?"?

both?Your program is POTENTIALLY input-bound because 62.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?36.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 