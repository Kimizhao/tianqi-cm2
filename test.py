from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
text_generation_zh = pipeline(Tasks.text_generation, model='/media/geoai/Data/nlp/output')
# result_zh = text_generation_zh("今日天气类型='浮尘'&空气质量等级='重度污染'&紫外线强度指数='中等'")
result_zh = text_generation_zh("今日天气类型='小雪'&预警等级='蓝色'&预警类型='道路冰雪'")
print(result_zh['text'])