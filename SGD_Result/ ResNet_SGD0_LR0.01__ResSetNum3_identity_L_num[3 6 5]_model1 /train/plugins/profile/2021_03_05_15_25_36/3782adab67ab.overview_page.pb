?	?n?oڻ|@?n?oڻ|@!?n?oڻ|@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?n?oڻ|@8??
j?s@1????Aa@A0.Ui?k??I6"??!@*	.????#?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorl#6@!Ұoc@?X@)l#6@1Ұoc@?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??辜??!rO??Y??)??辜??1rO??Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?qo~?D??!vY???4??)?מY???1( ?Tm??:Preprocessing2F
Iterator::Model@Û5x_??!9-?<???)??b??Հ?1??HJh??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap>?-:@!K????X@)ŏ1w-!o?1 ??s>޲?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 68.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???o??Q@Qal@n?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8??
j?s@8??
j?s@!8??
j?s@      ??!       "	????Aa@????Aa@!????Aa@*      ??!       2	0.Ui?k??0.Ui?k??!0.Ui?k??:	6"??!@6"??!@!6"??!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???o??Q@yal@n?=@?"7
model_5/conv2d_261/Conv2DConv2D??Qi朗?!??Qi朗?0"l
Bgradient_tape/model_5/batch_normalization_261/FusedBatchNormGradV3FusedBatchNormGradV3mn??g??!?????U??"7
model_5/conv2d_261/BiasAddBiasAdd?zy??0??!H???;??"h
<gradient_tape/model_5/conv2d_265/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX???ņ?!sf?/???0"f
;gradient_tape/model_5/conv2d_265/Conv2D/Conv2DBackpropInputConv2DBackpropInputǭ??d??!,<6?p׹?0"V
0model_5/batch_normalization_261/FusedBatchNormV3FusedBatchNormV3?ټ?u??!k??????"7
model_5/conv2d_267/Conv2DConv2D????ǅ?!%??b?Q??0"7
model_5/conv2d_273/Conv2DConv2DL??-???!???????0"7
model_5/conv2d_263/Conv2DConv2D?U??????!??i?`??0"7
model_5/conv2d_270/Conv2DConv2D???\???!9??6???0Q      Y@Y?va?????a&z6 ??X@qn?X?j)L@yG?\Bdp?"?

both?Your program is POTENTIALLY input-bound because 68.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?56.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 