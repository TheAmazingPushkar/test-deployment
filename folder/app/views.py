from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.shortcuts import redirect, render


def landing(request):
    if request.user.is_authenticated:
        return redirect('home')
    return render(request, 'app/landing.html')


def signup(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Signup successful. Welcome!')
            return redirect('home')
        messages.error(request, 'Please fix the errors below.')
    else:
        form = UserCreationForm()

    return render(request, 'app/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                return redirect('home')
        messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()

    return render(request, 'app/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('landing')


@login_required
def home(request):
    return render(request, 'app/home.html')

from django.shortcuts import render
from .ml.predictor import predict_image
from django.core.files.storage import FileSystemStorage

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']

        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        file_path = fs.path(filename)

        result = predict_image(file_path)

        return render(request, 'home.html', {
            'result': result,
            'image_url': fs.url(filename)
        })

    return render(request, 'home.html')