?	??R".?@??R".?@!??R".?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??R".?@?gǬ?v@1R?G?+b@A?,??????I`???%@*	?"?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatoru=?u?7$@!??~?X@)u=?u?7$@1??~?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchus??=A??!w7ůY??)us??=A??1w7ůY??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??vN?@??!&ؕ1x??)?HZ????1b*?Bz??:Preprocessing2F
Iterator::Model }??AѸ?!???c?b??)yxρ?y?1ȉ?ݼ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMape??9$@!#?8?:?X@)??z2??k?1?"?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIEd?+?Q@Q?nBQ?<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?gǬ?v@?gǬ?v@!?gǬ?v@      ??!       "	R?G?+b@R?G?+b@!R?G?+b@*      ??!       2	?,???????,??????!?,??????:	`???%@`???%@!`???%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qEd?+?Q@y?nBQ?<@?"9
model_33/conv2d_1783/Conv2DConv2D?#?)????!?#?)????0"n
Dgradient_tape/model_33/batch_normalization_1783/FusedBatchNormGradV3FusedBatchNormGradV3?L?VD???!?x*???"9
model_33/conv2d_1783/BiasAddBiasAddVQ?????!@/?l??"j
>gradient_tape/model_33/conv2d_1787/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterz????R??!???6??0"h
=gradient_tape/model_33/conv2d_1787/Conv2D/Conv2DBackpropInputConv2DBackpropInputA?NTd??!??=]???0"X
2model_33/batch_normalization_1783/FusedBatchNormV3FusedBatchNormV3kC??`??!$?? y???"9
model_33/conv2d_1795/Conv2DConv2D??E?4I??!????8??0"h
=gradient_tape/model_33/conv2d_1792/Conv2D/Conv2DBackpropInputConv2DBackpropInput"4????!???@p??0"9
model_33/conv2d_1785/Conv2DConv2Do??~7??!???%????0"9
model_33/conv2d_1792/Conv2DConv2D?????0??!??w???0Q      Y@Y??X????a\???`?X@q??/oD@y4???^?o?"?

both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 