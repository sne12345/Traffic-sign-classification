?	~?$A???@~?$A???@!~?$A???@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-~?$A???@ĳ?w@1??c?.?o@A?K?uT??I?*?WY?@*3^?I,??@)      p=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??7i$@!>k? ??X@)??7i$@1>k? ??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchE???????!k?02????)E???????1k?02????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism_}<?ݭ??!w??ea ??)g??e???11????^??:Preprocessing2F
Iterator::Model(E+????!)G?????)K>v()??1h@???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap6???'@!?^?9??X@)߿yq??m?1n-?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??Չ3?M@Qk*v?D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ĳ?w@ĳ?w@!ĳ?w@      ??!       "	??c?.?o@??c?.?o@!??c?.?o@*      ??!       2	?K?uT???K?uT??!?K?uT??:	?*?WY?@?*?WY?@!?*?WY?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??Չ3?M@yk*v?D@?"9
model_19/conv2d_1019/Conv2DConv2D?No???!?No???0"j
>gradient_tape/model_19/conv2d_1028/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?jG?>x??!?\?󪳩?0"j
>gradient_tape/model_19/conv2d_1021/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??b??!P2?lVr??0"j
>gradient_tape/model_19/conv2d_1025/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?3 `??!Ԡ?yV
??0"j
>gradient_tape/model_19/conv2d_1031/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^?'G??!???????0"h
=gradient_tape/model_19/conv2d_1027/Conv2D/Conv2DBackpropInputConv2DBackpropInput????*)??!꺊?@??0"h
=gradient_tape/model_19/conv2d_1024/Conv2D/Conv2DBackpropInputConv2DBackpropInput??	S??!Z?>F???0"h
=gradient_tape/model_19/conv2d_1030/Conv2D/Conv2DBackpropInputConv2DBackpropInputA??[??!?w???#??0"j
>gradient_tape/model_19/conv2d_1024/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX8<??E??!D;??h??0"j
>gradient_tape/model_19/conv2d_1030/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??b??A??!d?S2???0Q      Y@Y|?t8Y2??a,?6?X@q?j????H@y?5[?x`x?"?

both?Your program is POTENTIALLY input-bound because 58.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?49.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 