?	?	M?"}@?	M?"}@!?	M?"}@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?	M?"}@;?? *|t@1q??0c_@A??:?p??Inh?N?)@*	~?5^???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?\4d<?'@!???ʸX@)?\4d<?'@1???ʸX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??qQ-"??!??
?????)??qQ-"??1??
?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism
???ÿ?!???6?e??)?I+???1?Aj??A??:Preprocessing2F
Iterator::Model??&????!??S?$;??)?C4???y?1=!LҮ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??Y?r?'@!???l?X@)??+ٱq?1??#??D??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 70.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI,b-cDR@QO?wJs?:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?? *|t@;?? *|t@!;?? *|t@      ??!       "	q??0c_@q??0c_@!q??0c_@*      ??!       2	??:?p????:?p??!??:?p??:	nh?N?)@nh?N?)@!nh?N?)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q,b-cDR@yO?wJs?:@?"8
model_15/conv2d_801/Conv2DConv2D1V?ck??!1V?ck??0"m
Cgradient_tape/model_15/batch_normalization_801/FusedBatchNormGradV3FusedBatchNormGradV3?3	??,??!+?q?????"8
model_15/conv2d_801/BiasAddBiasAdd?X8,???! ?|<???"i
=gradient_tape/model_15/conv2d_805/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??%tb??!???dV??0"g
<gradient_tape/model_15/conv2d_805/Conv2D/Conv2DBackpropInputConv2DBackpropInput???|V2??!?@\?Wn??0"W
1model_15/batch_normalization_801/FusedBatchNormV3FusedBatchNormV3?_???'??!?&(?0??"8
model_15/conv2d_807/Conv2DConv2D?~9?k???!????????0"8
model_15/conv2d_803/Conv2DConv2DOn?)???!???|????0"8
model_15/conv2d_810/Conv2DConv2DW??5ދ?!???ϋl??0"g
<gradient_tape/model_15/conv2d_803/Conv2D/Conv2DBackpropInputConv2DBackpropInput???r?͋?!ʫ?k)??0Q      Y@YOG9?t??a??k,?X@qH??^??@@y1?%t?"?

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
Refer to the TF2 Profiler FAQb?33.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 