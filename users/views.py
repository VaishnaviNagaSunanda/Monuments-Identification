from django.http import JsonResponse
from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Monuments_Identification

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            pwd = form.cleaned_data.get('password')
            cpwd = form.cleaned_data.get('confirm_password')
            if pwd == cpwd:
                print('Data is Valid')
                form.save()
                messages.success(request, 'You have been successfully registered')
                form = UserRegistrationForm()
                return render(request, 'UserRegistrations.html', {'form': form})
            else:
                messages.error(request, 'Passwords do not match')
                return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Form is invalid or Email/Mobile Already Exists')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def terms_of_service(request):
    return render(request, 'terms_of_service.html')

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def training(request):
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
    from django.shortcuts import render
    from django.conf import settings as dj_settings

    # ---------- Check dataset availability before training ----------
    if request.method == "GET":
        # Check if model already exists
        model_save_path = os.path.join("models", "trained_model.h5")
        if os.path.exists(model_save_path):
            return render(request, "users/training_status.html", {
                "status": "Ready",
                "model_name": "Deep Learning CNN (MobileNetV2 derivative)",
                "classes_count": len(class_names),
                "last_modified": dt.datetime.fromtimestamp(os.path.getmtime(model_save_path)).strftime('%Y-%m-%d %H:%M:%S')
            })

    if request.method == "POST":
        train_dir = os.path.join(dj_settings.MEDIA_ROOT, "Indian-monuments", "images", "train")
        if not os.path.exists(train_dir):
            return render(request, "users/training_status.html", {
                "error": "Training dataset not found. Please ensure the dataset is present to re-train."
            })

        # ---------- Plot Function ----------
        def plot_loss_curves(history, save_path):
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            accuracy = history.history['accuracy']
            val_accuracy = history.history['val_accuracy']
            epochs = range(len(loss))

            plt.figure(figsize=(10,4))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, loss, label='Training Loss')
            plt.plot(epochs, val_loss, label='Validation Loss')
            plt.legend()
            plt.title('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(epochs, accuracy, label='Training Accuracy')
            plt.plot(epochs, val_accuracy, label='Validation Accuracy')
            plt.legend()
            plt.title('Accuracy')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        # ---------- Dataset Paths ----------
        train_dir = os.path.join(settings.MEDIA_ROOT, "Indian-monuments", "images", "train")
        test_dir  = os.path.join(settings.MEDIA_ROOT, "Indian-monuments", "images", "test")

        # ---------- Data Generators ----------
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        test_data = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        # ---------- CNN Model ----------
        model = Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            Conv2D(10, 3, activation='relu'),
            MaxPool2D(),
            Conv2D(10, 3, activation='relu'),
            MaxPool2D(),
            Flatten(),
            Dense(len(train_data.class_indices), activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # ---------- Train the Model ----------
        history = model.fit(
            train_data,
            epochs=5,
            steps_per_epoch=len(train_data),
            validation_data=test_data,
            validation_steps=len(test_data)
        )

        # ---------- Save Model in New Format (.h5) ----------
        os.makedirs("models", exist_ok=True)
        model_save_path = os.path.join("models", "trained_model.h5")
        model.save(model_save_path)

        # ---------- Save Training Curve ----------
        plot_path = os.path.join("models", "training_plot.png")
        plot_loss_curves(history, plot_path)

        # ---------- Context to HTML ----------
        context = {
            "accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1],
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "plot_path": plot_path,
            "model_path": model_save_path
        }

        return render(request, "users/training_result.html", context)

    # Fallback to status page
    return render(request, "users/training_status.html", {})



import tensorflow as tf
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing import image


# List of monument class names (must match training order)
class_names = [
    'Ajanta Caves', 'Chhota_Imambara', 'alai_darwaza', 'alai_minar',
    'basilica_of_bom_jesus', 'charminar', 'golden_temple',
    'tajmahal', 'tanjavur temple', 'victoria memorial'
]

# Dictionary mapping monument names to info and AR resources
monument_info = {
    "Ajanta Caves": {
        "history": "The Ajanta Caves are 30 rock-cut Buddhist cave monuments in Maharashtra.",
        "3d_model_url": "/static/ar_models/ajanta.glb",
        "map_location": "https://maps.google.com/?q=Ajanta+Caves"
    },
    "alai_darwaza": {
        "history": "Built in 1311, the Alai Darwaza is the southern gateway of the Quwwat-ul-Islam Mosque in Delhi.",
        "3d_model_url": "/static/ar_models/alai_darwaza.glb",
        "map_location": "https://maps.google.com/?q=Alai+Darwaza"
    },
    "alai_minar": {
        "history": "An unfinished tower in the Qutb complex started by Alauddin Khalji.",
        "3d_model_url": "/static/ar_models/alai_minar.glb",
        "map_location": "https://maps.google.com/?q=Alai+Minar"
    },
    "basilica_of_bom_jesus": {
        "history": "UNESCO World Heritage Site in Goa, holds the remains of St. Francis Xavier.",
        "3d_model_url": "/static/ar_models/bom_jesus.glb",
        "map_location": "https://maps.google.com/?q=Basilica+of+Bom+Jesus"
    },
    "charminar": {
        "history": "Iconic 16th-century mosque in Hyderabad built by Muhammad Quli Qutb Shah.",
        "3d_model_url": "/static/ar_models/charminar.glb",
        "map_location": "https://maps.google.com/?q=Charminar"
    },
    "Chhota_Imambara": {
        "history": "Historical monument in Lucknow built by Muhammad Ali Shah in 1838.",
        "3d_model_url": "/static/ar_models/chhota_imambara.glb",
        "map_location": "https://maps.google.com/?q=Chhota+Imambara"
    },
    "golden_temple": {
        "history": "The holiest Gurdwara of Sikhism located in Amritsar.",
        "3d_model_url": "/static/ar_models/golden_temple.glb",
        "map_location": "https://maps.google.com/?q=Golden+Temple"
    },
    "tajmahal": {
        "history": "Famous white marble mausoleum built by Shah Jahan in Agra.",
        "3d_model_url": "/static/ar_models/tajmahal.glb",
        "map_location": "https://maps.google.com/?q=Taj+Mahal"
    },
    "tanjavur temple": {
        "history": "Known for its grand architecture, Brihadeeswarar Temple is a UNESCO World Heritage site.",
        "3d_model_url": "/static/ar_models/tanjavur_temple.glb",
        "map_location": "https://maps.google.com/?q=Tanjavur+Temple"
    },
    "victoria memorial": {
        "history": "Large marble building in Kolkata, built in honor of Queen Victoria.",
        "3d_model_url": "/static/ar_models/victoria_memorial.glb",
        "map_location": "https://maps.google.com/?q=Victoria+Memorial"
    },
}


from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from django.conf import settings as django_settings

# Load the model ONCE at startup so each request doesn't reload it (fixes timeout)
_MONUMENT_MODEL = None

def get_model():
    global _MONUMENT_MODEL
    if _MONUMENT_MODEL is None:
        model_path = os.path.join(django_settings.BASE_DIR, 'models', 'trained_model.h5')
        _MONUMENT_MODEL = tf.keras.models.load_model(model_path, compile=False)
    return _MONUMENT_MODEL


@csrf_exempt
def prediction(request):

    # If user opens page in browser
    if request.method == "GET":
        return render(request, "users/monument_prediction.html")

    # If image uploaded
    if request.method == "POST" and 'monument_image' in request.FILES:

        try:
            import tempfile

            # Use cached model (loaded once at startup)
            model = get_model()

            uploaded_file = request.FILES['monument_image']

            # Save to /tmp which is always writable (even on Hugging Face)
            tmp_dir = tempfile.gettempdir()
            fs = FileSystemStorage(location=tmp_dir, base_url='/tmp/')
            file_path = fs.save(uploaded_file.name, uploaded_file)
            full_path = fs.path(file_path)

            # Preprocess image
            img = image.load_img(full_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_batch)
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100

            # Increased threshold for better reliability (e.g. to avoid false positives like Lotus Temple -> Taj Mahal)
            THRESHOLD = 75

            if confidence < THRESHOLD:
                predicted_class = "Invalid Image"
                history = "This image does not belong to the trained monuments dataset."
                model_3d = ""
                map_link = ""
            else:
                predicted_class = class_names[predicted_index]
                info = monument_info.get(predicted_class, {})
                history = info.get("history", "Information not available.")
                model_3d = info.get("3d_model_url", "")
                map_link = info.get("map_location", "")

            confidence = round(confidence, 2)

            # Build a data URL for the uploaded image so it displays correctly
            import base64
            with open(full_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            ext = uploaded_file.name.split('.')[-1].lower()
            mime = 'image/jpeg' if ext in ['jpg', 'jpeg'] else f'image/{ext}'
            uploaded_file_url = f"data:{mime};base64,{img_data}"

            # If request comes from mobile app
            if "application/json" in request.headers.get("Accept", ""):
                return JsonResponse({
                    "uploaded_image": uploaded_file_url,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "history": history,
                    "model_3d": model_3d,
                    "map_link": map_link
                })

            # If request comes from browser form
            return render(request, "users/monument_prediction.html", {
                "uploaded_file_url": uploaded_file_url,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "history": history,
                "model_3d": model_3d,
                "map_link": map_link,
            })

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            # Always show error detail so we can diagnose HF issues
            return render(request, "users/monument_prediction.html", {"error": error_detail})

    return JsonResponse({"error": "Invalid request"})






# ---------- Mobile API Prediction ----------
@csrf_exempt
def api_predict(request):

    if request.method == 'POST' and 'monument_image' in request.FILES:

        from django.conf import settings
        import os
        model = get_model()

        uploaded_file = request.FILES['monument_image']
        import tempfile
        tmp_dir = tempfile.gettempdir()
        fs = FileSystemStorage(location=tmp_dir, base_url='/tmp/')
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        img = image.load_img(full_path, target_size=(224, 224))
        img_array = image.img_to_array(img)/255.0
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch)

        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        THRESHOLD = 75
        if confidence < THRESHOLD:
            predicted_class = "Invalid Image"
            history = "This image does not belong to the trained monuments dataset."
            model_3d = ""
            map_link = ""
        else:
            predicted_class = class_names[predicted_index]
            info = monument_info.get(predicted_class, {})
            history = info.get("history", "")
            map_link = info.get("map_location", "")
            model_3d = info.get("3d_model_url", "")

        return JsonResponse({
            "uploaded_image": fs.url(file_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence,2),
            "history": history,
            "map_link": map_link,
            "model_3d": model_3d
        })

    return JsonResponse({"error":"Invalid request"})
