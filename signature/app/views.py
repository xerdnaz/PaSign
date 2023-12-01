import os
from django.views import View
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from .forms import UserDataForm
from django.http import JsonResponse
from .models import UserData, get_user_signature_path
from django.contrib import messages
from django.conf import settings

# LANDING PAGE
class LandingPageView(View):
    landing_page = 'index.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.landing_page)

# LOGIN
class LoginPageView(View):
    login_page = 'landing/login-index.html'

    def get(self, request):
        return render(request, self.login_page)

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')  # Check if the checkbox is checked

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            if not remember_me:
                # If "Remember Me" is not checked, set session expiry to 0 (browser session)
                request.session.set_expiry(0)

            return redirect('dashboard_page')  # Replace with the actual URL name for your dashboard page
        else:
            # Authentication failed, handle accordingly (e.g., show an error message)
            error_message = 'Invalid username or password. Please try again.'
            messages.error(request, error_message)
            return render(request, self.login_page, {'error_message': error_message})

        return render(request, self.login_page)
        
# FORGOT PASSWORD
class ForgotPassView(View):
    forgotpassword = 'forgotpass.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.forgotpassword)

# DASHBOARD
class DashboardView(View):
    dashboard_page = 'dashboard.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.dashboard_page)
    
# UPLOAD
class UploadView(View):
    upload_data = 'upload.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.upload_data)
    
# REGISTER STUDENT
def register_user(request):
    if request.method == 'POST':
        form = UserDataForm(request.POST, request.FILES)

        signatures = request.FILES.getlist('signature_files')

        if form.is_valid() and len(signatures) == 3:
            # Check for existing user with case-insensitive comparison
            student_id = form.cleaned_data['student_id']
            email = form.cleaned_data['email'].lower()

            # Check if a user with the same student ID or email already exists
            if UserData.objects.filter(student_id=student_id).exclude(id=form.instance.id).exists() or \
               UserData.objects.filter(email__iexact=email).exclude(id=form.instance.id).exists():
                messages.error(request, 'User with similar information already exists. Please check and try again.')
                return render(request, 'upload.html', {'form': form})

            user_data = form.save()

            # Use the get_user_signature_path function to determine the path
            signature_folder = get_user_signature_path(user_data, '')


            for i, signature_file in enumerate(signatures, start=1):
                file_extension = os.path.splitext(signature_file.name)[1]
                new_file_name = f'{student_id}_signature_{i}{file_extension}'
                new_file_path = os.path.join(settings.MEDIA_ROOT, signature_folder, new_file_name)
                
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

                with open(new_file_path, 'wb') as file:
                    file.write(signature_file.read())
                
                setattr(user_data, f'signature_{i}', os.path.join(signature_folder, new_file_name))

            user_data.save()

            # Add success message
            messages.success(request, 'Registration successful!')

            return redirect('upload_data')  # Redirect after successful registration

    else:
        form = UserDataForm()

    return render(request, 'upload.html', {'form': form})


# TEST
class TestView(View):
    test_data = 'test.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.test_data)

def get_student_data(request, student_id):
    try:
        student = UserData.objects.get(student_id=student_id)
        data = {
            'name': f"{student.first_name} {student.last_name}",
            'email': student.email,
            'signature_1_url': student.signature_1.url if student.signature_1 else '',
            'signature_2_url': student.signature_2.url if student.signature_2 else '',
            'signature_3_url': student.signature_3.url if student.signature_3 else '',
        }
        return JsonResponse(data)
    except UserData.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)
    
#MODEL