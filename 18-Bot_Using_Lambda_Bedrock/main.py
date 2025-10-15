import boto3
from datetime import datetime
import json
client = boto3.client('bedrock-runtime')
def generateResponse(topic:str)->str:
    conversation = [
        {
            'role':'assistant',
            'content':[{'text':'Your a bot which is expert in blog creation , for the input topic create a bot which should not be more than 500 words'}]
        },
        {
            'role':'user',
            'content':[{'text':topic}]
        }
    ]

    response = client.converse(modelId = 'qwen.qwen3-coder-30b-a3b-v1:0',messages = conversation)
    return response['output']['message']['content'][0]['text']

def saveResponseToS3(blog:str,topic:str):
    s3_key = f"BlogOutput/{topic}{datetime.now().time()}.pdf"
    s3_bucket = 'BlogGeneration'
    s3Bucket = boto3.client('s3')
    response = s3Bucket.putObject(Bucket = s3_bucket,Key = s3_key , Body = blog)
    print('Put Successful',response)

def lambda_handler(event, context):
    # TODO implement
    event=json.loads(event["body"])
    blogTopic = event['BlogTopic']
    response = generateResponse(blogTopic)
    if response:
        saveResponseToS3(response,blogTopic)
        return {
        'statusCode': 201,
        'body': json.dumps('Blog Created and successful saved to S3')
        }
    else:
        print('Blog Not Generated')
        return {
        'statusCode': 400,
        'body': json.dumps('NOT ABLE TO CREATE BLOG CHECK LOGS')
        }

    