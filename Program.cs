using BurnAnalysisApp;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.FileProviders;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Microsoft.AspNetCore.StaticFiles;


var builder = WebApplication.CreateBuilder(args);

// JWT Ayarlarını Al
var jwtSettings = builder.Configuration.GetSection("JwtSettings");
var key = Encoding.UTF8.GetBytes(jwtSettings["Key"]!);

// JWT Servisini Ekle
builder.Services.AddScoped<IJwtService, JwtService>();

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = new SymmetricSecurityKey(key),
            ValidateIssuer = true,
            ValidIssuer = jwtSettings["Issuer"],
            ValidateAudience = true,
            ValidAudience = jwtSettings["Audience"],
            ValidateLifetime = true,
            ClockSkew = TimeSpan.Zero // Token süresi dolduğunda anında geçersiz olması için
        };
    });

// PostgreSQL DbContext Bağlantısı
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

// Servis Bağımlılıklarını Kaydet
builder.Services.AddScoped<IEmailService, EmailService>();

// CORS Politikası Ekle
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowReactApp", policy =>
    {
        policy.WithOrigins("http://localhost:3000") // React uygulamasının olduğu port
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials(); // CORS üzerinden kimlik doğrulaması yapılacaksa AllowCredentials kullanılır.
    });
});

// API Servislerini Ekle
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddHostedService<ReminderBackgroundService>();

var app = builder.Build();

// Ortam Konfigürasyonu
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors("AllowReactApp"); // CORS politikasını uygula
app.UseHttpsRedirection();
app.UseAuthentication();  // JWT doğrulamasını uygula
app.UseAuthorization();

// MIME Türlerini Ayarla
var provider = new FileExtensionContentTypeProvider();
provider.Mappings[".mp4"] = "audio/mp4"; // **.mp4 olarak değiştirildi**


app.UseStaticFiles(new StaticFileOptions
{
    ContentTypeProvider = provider
});

app.MapControllers();
//burası yeni
using (var scope = app.Services.CreateScope())
{
    var context = scope.ServiceProvider.GetRequiredService<AppDbContext>();

    var notifications = await context.Notifications
        .Where(n => n.ForumPostID == null)
        .ToListAsync();

    foreach (var notification in notifications)
    {
        // Doktoru veritabanından çek
        var doctor = await context.Doctors
            .FirstOrDefaultAsync(d => d.DoctorID == notification.DoctorID);

        if (doctor != null)
        {
            // Doktorun forum postunu bul
            var relatedPost = await context.ForumPosts
                .FirstOrDefaultAsync(fp => fp.DoctorName == doctor.Name);

            if (relatedPost != null)
            {
                notification.ForumPostID = relatedPost.ForumPostID;
            }
        }
    }

    await context.SaveChangesAsync();
}



app.Run();