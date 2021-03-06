?	I?p??@I?p??@!I?p??@	?ߕ????ߕ???!?ߕ???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I?p??@??V_]`s@1?Z?̕g@AϢw*????I䠄??1@YϿ]??N??*	F????ÿ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator]?gA( @!F?????X@)]?gA( @1F?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???l ??!I rA?S??)???l ??1I rA?S??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`[??gͷ?!jח?^K??)??????1{????:Preprocessing2F
Iterator::Modelj?L?:??!?.cVd??)??;???v?1??q?|???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapΨ?*? @!,Ds?n?X@)L??m?1?.wrGU??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?ߕ???I?Fp??O@Qd??	?HB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??V_]`s@??V_]`s@!??V_]`s@      ??!       "	?Z?̕g@?Z?̕g@!?Z?̕g@*      ??!       2	Ϣw*????Ϣw*????!Ϣw*????:	䠄??1@䠄??1@!䠄??1@B      ??!       J	Ͽ]??N??Ͽ]??N??!Ͽ]??N??R      ??!       Z	Ͽ]??N??Ͽ]??N??!Ͽ]??N??b      ??!       JGPUY?ߕ???b q?Fp??O@yd??	?HB@?"9
model_24/conv2d_1293/Conv2DConv2DG^ ????!G^ ????0"j
>gradient_tape/model_24/conv2d_1302/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?F`???!?@?JF??0"j
>gradient_tape/model_24/conv2d_1299/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^?? 8??!??P?J???0"j
>gradient_tape/model_24/conv2d_1295/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??oś?!?_????0"h
=gradient_tape/model_24/conv2d_1301/Conv2D/Conv2DBackpropInputConv2DBackpropInput??\???!z[g???0"h
=gradient_tape/model_24/conv2d_1298/Conv2D/Conv2DBackpropInputConv2DBackpropInput?9)R}???!?A0?|??0"j
>gradient_tape/model_24/conv2d_1301/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)?-|?U??! ??L2??0"9
model_24/conv2d_1299/Conv2DConv2D??n8??!o?j	????0"9
model_24/conv2d_1295/Conv2DConv2Dk #????!v=?t??0"9
model_24/conv2d_1302/Conv2DConv2D?^????!c??????0Q      Y@Y6?NK?~??ag????X@q~Fz??UD@y?Rʮi?"?
both?Your program is POTENTIALLY input-bound because 60.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 