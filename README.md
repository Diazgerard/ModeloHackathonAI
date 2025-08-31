# **Agente de Validación de Comentarios**

Este proyecto corresponde al **agente de IA** que valida los comentarios de los estudiantes antes de almacenarlos en la blockchain. Su función es evaluar la seguridad de los textos y, en caso necesario, sugerir reformulaciones para mantener un espacio respetuoso y constructivo.

---

## **Requisitos previos**

* **Python 3.8 - 3.11** instalado en tu sistema.

---

## **Configuración**

1. Clonar el repositorio:

```bash
git clone https://github.com/Diazgerard/ModeloHackathonAI
cd ModeloHackathonAI
```

2. Crear y activar un entorno virtual:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

4. Copiar el archivo `.env_template` a `.env` y agregar tu API Key de Groq:

```env
GROQ_API_KEY="your_api_key_here"
```

Puedes crear una cuenta y obtener tu API Key en: [https://console.groq.com/keys](https://console.groq.com/keys)

---

## **Ejecutar el proyecto**

Inicia el servidor con:

```bash
python api.py
```

Esto levantará la aplicación en `http://localhost:5000`.

---

⚡ El agente quedará corriendo y listo para validar los comentarios enviados desde la DApp.