import os
import sys
import json
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configuración para Windows y PyTorch
if sys.platform == "win32":
    try:
        import torch
        torch.classes.__path__ = []
    except ImportError:
        pass  

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Cargar variables de entorno
load_dotenv()

# Configurar Flask
app = Flask(__name__)
CORS(app) 

# Configurar claves API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: No se encontró la clave API de Groq. Asegúrate de configurar GROQ_API_KEY en tu archivo .env")
    sys.exit(1)

model = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

def load_tags():
    """Cargar tags desde el archivo tags.txt"""
    try:
        with open("tags.txt", "r", encoding="utf-8") as file:
            tags = [line.strip().lower() for line in file if line.strip()]
        return tags
    except FileNotFoundError:
        print("Warning: No se encontró el archivo tags.txt")
        return []
    except Exception as e:
        print(f"Error cargando tags: {str(e)}")
        return []

def is_coherent_text(text):
    """Validar si el texto es coherente (más de una palabra y tiene sentido básico)"""
    text = text.strip()
    
    # 1. Verificar que no esté vacío
    if not text:
        return False
    
    # 2. Verificar que tenga al menos 2 palabras
    words = text.split()
    if len(words) < 2:
        return False
    
    # 3. Verificar que no sean solo caracteres especiales o números
    clean_text = re.sub(r'[^\w\s]', '', text)
    if len(clean_text.strip()) < 3:
        return False
    
    # 4. Verificar que no sean solo repeticiones de la misma palabra
    unique_words = set(word.lower() for word in words if word.isalpha())
    if len(unique_words) < 2:
        return False
    
    # 5. Detectar incoherencia semántica usando reglas híbridas
    # Lista de patrones específicamente incoherentes
    incoherent_patterns = [
        "casa azul mojado", "perro volando matemáticas", "mesa correr feliz",
        "computadora cantar verde", "silla bailar número", "árbol escribir calor",
        "teléfono dormir azúcar", "libro nadar rojo", "ventana comer fríos",
        "zapato volar música", "reloj bailar agua", "puerta correr números"
    ]
    
    # Si es exactamente uno de estos casos, es incoherente
    if text.lower().strip() in incoherent_patterns:
        return False
    
    # Verificar patrones de incoherencia (sustantivo + verbo incongruente + adjetivo/sustantivo)
    # Ejemplo: "casa cantar azul" (objeto físico + acción incompatible + descriptor)
    if len(words) == 3:
        # Objetos físicos que no pueden realizar ciertas acciones
        objects = {'casa', 'mesa', 'silla', 'puerta', 'ventana', 'libro', 'teléfono', 'computadora'}
        impossible_actions = {'cantar', 'bailar', 'correr', 'volar', 'nadar', 'dormir', 'comer'}
        
        word1, word2, word3 = [w.lower() for w in words]
        
        # Si primer palabra es objeto y segunda es acción imposible
        if word1 in objects and word2 in impossible_actions:
            return False
    
    # Verificar estructura mínima de oración en español
    structure_indicators = {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'es', 'está', 'son', 'están', 'tiene', 'tienen', 'hay', 'fue', 'era',
        'de', 'del', 'en', 'con', 'por', 'para', 'desde', 'hasta', 'sobre',
        'que', 'se', 'me', 'te', 'le', 'nos', 'les', 'mi', 'tu', 'su',
        'muy', 'más', 'menos', 'bien', 'mal', 'no', 'sí', 'y', 'o', 'pero',
        'quiero', 'necesito', 'creo', 'pienso', 'siento', 'veo', 'escucho',
        'necesitamos', 'queremos', 'podemos', 'debemos'
    }
    
    text_words = set(word.lower() for word in words)
    has_structure = bool(text_words.intersection(structure_indicators))
    
    if not has_structure:
        return False
    
    return True

def extract_tags_from_text(text, available_tags):
    """Extraer tags relevantes del texto - cantidad dinámica según longitud"""
    text_lower = text.lower()
    tag_scores = {}  
    # Palabras del texto
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    # Determinar cantidad máxima de tags según longitud del texto
    if len(words_in_text) <= 4:  
        max_tags = 1
    elif len(words_in_text) <= 8: 
        max_tags = 2
    else:  
        max_tags = 3
    
    for tag in available_tags:
        score = 0
        matches = []  
        
        # 1. Búsqueda exacta
        for i, word in enumerate(words_in_text):
            if word == tag:
                matches.append(i)
                score += 3  
        
        # 2. Búsqueda de variaciones simples
        if not matches: 
            for i, word in enumerate(words_in_text):
                if word == tag + 's' or tag == word + 's':
                    matches.append(i)
                    score += 2  
                    break
                elif word == tag + 'es' or tag == word + 'es':
                    matches.append(i)
                    score += 2
                    break
        
        # Si encontramos el tag, calcular puntuación final
        if matches:
            score += len(tag) * 0.1
            first_position = min(matches)
            position_bonus = max(0, 5 - first_position)
            score += position_bonus
            frequency_bonus = len(matches) * 0.5
            score += frequency_bonus
            tag_scores[tag] = score
    
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    most_relevant_tags = [tag for tag, score in sorted_tags[:max_tags]]
    
    return most_relevant_tags

def categorize_comment(comment):
    """Categorizar el comentario usando LLM"""
    template = """
    Analiza el siguiente comentario y categorizalo EXACTAMENTE en una de estas cuatro categorías:
    - "Sugerencia": Si el comentario propone mejoras, ideas, cambios o recomendaciones constructivas
    - "Opinion": Si el comentario expresa una opinión personal neutral o positiva, experiencias sin ser ofensivo
    - "Queja": Si el comentario contiene lenguaje ofensivo, discriminatorio, amenazas, insultos, críticas muy negativas, o sentimientos muy negativos hacia personas (ej: "el maestro es malo", "odio a...", "es terrible", etc.)
    - "Vida universitaria": Si el comentario se refiere específicamente a experiencias, situaciones, actividades o aspectos de la vida universitaria, académica o estudiantil que no encajan en las otras categorías

    Reglas importantes:
    1. Responde SOLO con una de estas cuatro palabras: "Sugerencia", "Opinion", "Queja", o "Vida universitaria"
    2. No agregues explicaciones adicionales
    3. Comentarios negativos sobre personas (maestros, compañeros, etc.) van en "Queja"
    4. Comentarios sobre clases, universidad, estudios, campus, etc. van en "Vida universitaria"
    5. Si hay duda, prioriza en este orden: Queja > Vida universitaria > Sugerencia > Opinion

    Ejemplos:
    - "El maestro es malo" → Queja
    - "La clase de matemáticas es difícil" → Vida universitaria
    - "Deberían mejorar la cafetería" → Sugerencia
    - "Me gusta estudiar" → Opinion

    Comentario: "{comment}"
    
    Categoría:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"comment": comment}):
            response += chunk.content
        
        # Limpiar la respuesta y validar
        category = response.strip()
        valid_categories = ["Sugerencia", "Opinion", "Queja", "Vida universitaria"]
        
        if category in valid_categories:
            return category
        else:
            for valid_cat in valid_categories:
                if valid_cat.lower() in category.lower():
                    return valid_cat
            return "Opinion"
            
    except Exception as e:
        print(f"Error categorizando comentario: {str(e)}")
        return "Opinion"

def formalize_hate_speech(comment):
    """Convertir comentario ofensivo a lenguaje formal y apropiado"""
    template = """
    El siguiente comentario contiene lenguaje ofensivo. Conviértelo a un comentario formal, respetuoso y constructivo que exprese la misma idea pero de manera apropiada para un entorno académico o profesional.

    Reglas:
    1. Eliminar todas las palabras ofensivas, vulgaridades o insultos
    2. Mantener la esencia del mensaje pero en tono constructivo
    3. Usar lenguaje formal y respetuoso
    4. Si es una queja, convertirla en feedback constructivo
    5. Máximo 2-3 líneas
    6. Responder SOLO con el texto formalizado, sin explicaciones adicionales

    Comentario original: "{comment}"

    Comentario formalizado:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"comment": comment}):
            response += chunk.content
        
        formalized = response.strip()
        
        if formalized.startswith('"') and formalized.endswith('"'):
            formalized = formalized[1:-1]
        elif formalized.startswith("'") and formalized.endswith("'"):
            formalized = formalized[1:-1]
        
        formalized = formalized.replace('""', '').replace("''", '')
        
        if len(formalized) < 10:
            formalized = "Comentario convertido a lenguaje apropiado por contener contenido ofensivo."
        
        return formalized
        
    except Exception as e:
        print(f"Error formalizando comentario: {str(e)}")
        return "Comentario modificado por contener contenido inapropiado."

def save_to_json(data, filename="comentarios_analizados.json"):
    """Guardar datos en archivo JSON"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        
        existing_data.append(data)
        
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error guardando en JSON: {str(e)}")
        return False

def load_analysis_history():
    """Cargar historial de análisis desde JSON"""
    try:
        if os.path.exists("comentarios_analizados.json"):
            with open("comentarios_analizados.json", "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        print(f"Error cargando historial: {str(e)}")
        return []

# Cargar tags disponibles
available_tags = load_tags()


# RUTAS DE LA API

# Variable global para almacenar el comentario actual
comentario_actual = None

@app.route('/comentario', methods=['GET'])
def obtener_comentario():
    """Endpoint para obtener el comentario actual (del momento)"""
    global comentario_actual
    
    if comentario_actual is None:
        return jsonify({
            "error": "No hay comentario disponible en este momento"
        }), 404
    
    return jsonify({
        "success": True,
        "data": {
            "comentario": comentario_actual
        }
    })

@app.route('/comentario', methods=['POST'])
def enviar_analisis():
    """Endpoint para enviar el análisis del comentario (comentario + categoría + tags)"""
    global comentario_actual
    
    try:
        data = request.get_json()
        
        required_fields = ['comentario', 'categoria', 'tags']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Faltan los siguientes campos requeridos: {', '.join(missing_fields)}"
            }), 400
        
        comment = data['comentario'].strip()
        categoria = data['categoria'].strip()
        tags = data['tags']
        
        if not comment:
            return jsonify({
                "error": "El comentario no puede estar vacío"
            }), 400
        
        valid_categories = ["Sugerencia", "Opinion", "Queja", "Vida universitaria"]
        if categoria not in valid_categories:
            return jsonify({
                "error": f"Categoría inválida. Categorías válidas: {valid_categories}"
            }), 400
        
        if not isinstance(tags, list):
            return jsonify({
                "error": "Los tags deben ser una lista"
            }), 400
        
        analysis_data = {
            "id": len(load_analysis_history()) + 1,
            "timestamp": datetime.now().isoformat(),
            "comentario": comment,
            "categoria": categoria,
            "tags": tags
        }
        
        if save_to_json(analysis_data):
            comentario_actual = None
            
            return jsonify({
                "success": True,
                "data": analysis_data,
                "message": "Análisis del comentario guardado exitosamente"
            })
        else:
            return jsonify({
                "error": "Error al guardar el análisis"
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}"
        }), 500



# Manejo de errores globales
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Método no permitido para este endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Error interno del servidor"
    }), 500

if __name__ == '__main__':
    # Ejecutar la aplicación
    app.run(debug=True, host='0.0.0.0', port=5000)