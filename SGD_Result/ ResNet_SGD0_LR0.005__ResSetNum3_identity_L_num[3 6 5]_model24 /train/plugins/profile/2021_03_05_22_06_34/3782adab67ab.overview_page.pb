?	?h?hs0?@?h?hs0?@!?h?hs0?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?h?hs0?@?S??v@16???Вb@A?N@a???I~nh?NO@*?z????@)      p=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?$?9?@!??]'C[X@)?$?9?@1??]'C[X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch&?2????!??1?>n @)&?2????1??1?>n @:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?s????!::???=@)
?8?*??1?D?U?z??:Preprocessing2F
Iterator::Model??F???!`?Yٟ@)
??O?my?1??mh??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??KU??@!?25S_X@)??7?ܘn?1z??\g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?%]?Q@Q]??k??<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S??v@?S??v@!?S??v@      ??!       "	6???Вb@6???Вb@!6???Вb@*      ??!       2	?N@a????N@a???!?N@a???:	~nh?NO@~nh?NO@!~nh?NO@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?%]?Q@y]??k??<@?"9
model_28/conv2d_1502/Conv2DConv2D~?ދH???!~?ދH???0"n
Dgradient_tape/model_28/batch_normalization_1502/FusedBatchNormGradV3FusedBatchNormGradV3SԝW????!I8??{???"9
model_28/conv2d_1502/BiasAddBiasAdd??Ѻ2s??!?s=?ݳ?"j
>gradient_tape/model_28/conv2d_1506/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter; ?IKO??!?yq?Ƕ?0"h
=gradient_tape/model_28/conv2d_1506/Conv2D/Conv2DBackpropInputConv2DBackpropInputbl=?F???!T?@F???0"X
2model_28/batch_normalization_1502/FusedBatchNormV3FusedBatchNormV3?3軔&??!?ǽ??k??"9
model_28/conv2d_1504/Conv2DConv2D?H??Q??!??{?/??0"9
model_28/conv2d_1511/Conv2DConv2DG?'????!=ep?u???0"9
model_28/conv2d_1508/Conv2DConv2Dm@ba???!D??m?Z??0"9
model_28/conv2d_1514/Conv2DConv2D?}B??! A????0Q      Y@Y?va?????a&z6 ??X@q??]??@J@y??ѣ?:p?"?

both?Your program is POTENTIALLY input-bound because 69.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?52.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 