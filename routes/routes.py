from domain.domain import Request,Response
from service.service import chat
from flask import request,Blueprint,jsonify

chat_bp=Blueprint('chat_bp',__name__)

@chat_bp.route('/chat',methods=['POST'])
def translate():
    print(request.json())
    try:
        data=request.json()
        print("data: ",data)
        if not data:
            return jsonify({
                "message":"error from client"
            }),400
        
        request=Request.from_dict(data=data)
        response=chat(request.prompt)
        response=Response(response=response)

        return jsonify({
            response.to_dict()
        }),200
    except ValueError as ve:
        return jsonify({
            "message":str(ve),
            "error":"value error"
        }),400
    except Exception as e:
        return jsonify({
            "message":str(e),
            "error":"server error"
        }),500
    
        